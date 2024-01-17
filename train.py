import os
import argparse
import random
import torchvision
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
from scripts.demo.turbo import *
from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from pytorch_lightning import seed_everything
from mmdet.apis import init_detector, inference_detector
from utils import chunk, get_rand, IoU, load_classes
from seg_module import Segmodule
from evaluate import evaluate

warnings.filterwarnings("ignore")



def main(args):
    
    seed_everything(args.seed)

    class_train, class_test, class_coco = load_classes(args.class_split)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_file = 'src/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
    pretrain_detector = init_detector(config_file, checkpoint_file, device=device)
    
    seg_module = Segmodule().to(device)
    # seg_module = torch.compile(seg_module)

    version_dict = VERSION2SPECS["SDXL-Turbo"]
    state = init_st(version_dict, load_filter=True)
    model = state["model"]
    load_model(model)

    sampler = SubstepSampler(
        n_sample_steps=args.n_steps,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    sampler.noise_sampler = SeededNoise(seed=args.seed)

    os.makedirs(args.exp_dir, exist_ok=True)
    img_dir = os.path.join(args.exp_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.exp_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'logs'))
    
    learning_rate = 2e-5
    total_iter = 30000

    g_optim = optim.Adam(
        [{"params": seg_module.parameters()},],
        lr=learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=15000, gamma=0.5)
    
    class_embedding_dic = {}
    for class_name in class_train:
        class_embedding, uc = get_cond(model, H=args.H, W=args.W, prompt=class_name)
        class_embedding = class_embedding['crossattn'][:, 1, :].unsqueeze(1)
        class_embedding_dic[class_name] = class_embedding

    print('***********************   begin   **********************************')
    print(f"Start training with maximum {total_iter} iterations.")
    
    total_loss = 0
    total_iou = 0
    
    for j in tqdm(range(1, total_iter+1)):
        lr_scheduler.step()
        if not args.from_file:
            trainclass = class_train[random.randint(0, len(class_train)-1)]
            prompt = "A scene containing a" + trainclass
        # if not args.from_file:
        #     trainclass = class_train[random.randint(0, len(class_train)-1)]
        #     otherclass = class_train[random.randint(0, len(class_train)-1)]
        #     rand = random.random()
        #     if rand >= 0.5: prompt = f"a photograph of a {trainclass} and a {otherclass}."
        #     else: prompt = f"a photograph of a {otherclass} and a {trainclass}."
        else:
            print(f"reading prompts from {args.from_file}")
            with open(args.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(data)
        # class_index = class_coco[trainclass]
        
        # generate images
        seed = get_rand()
        out = sample(
            model, sampler, H=args.H, W=args.W, seed=seed, 
            prompt=prompt, filter=state.get("filter")
        )
        
        # detector
        result = inference_detector(pretrain_detector, out[0])
        flag = True # detect if mmdet fail to detect the object
        for instance in result.pred_instances:
            if instance.labels[0] != class_coco[trainclass]: continue
            if instance.scores[0] < 0.7: continue
            gt_seg = instance.masks[0]
            flag = False
            break
        if flag: continue # "pretrain detector fail to detect the object
        gt_seg = gt_seg.unsqueeze(0).float() # 1, 512, 512
        
        # seg_module
        pred_seg = seg_module(get_feature_dic(), class_embedding_dic[trainclass])
        pred_seg = pred_seg.squeeze(0)
    
        loss = F.binary_cross_entropy_with_logits(pred_seg, gt_seg)
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        total_loss += loss.item()
        
        pred_seg = torch.sigmoid(pred_seg)
        pred_seg[pred_seg <= 0.5] = 0
        pred_seg[pred_seg > 0.5] = 1
        iou = IoU(pred_seg, gt_seg)
        total_iou += iou
        
        # visualization 
        if  j % 100 == 0:
            writer.add_scalar('train/loss', total_loss/100, global_step=j)
            writer.add_scalar('train/iou', total_iou/100, global_step=j)
            total_loss = 0
            total_iou = 0
    
            viz = torch.cat([gt_seg, pred_seg], axis=1)
            torchvision.utils.save_image(viz, 
                            img_dir +'/viz_sample_{0:05d}_seg'.format(j)+trainclass+'.png', 
                            normalize=True, scale_each=True)
            Image.fromarray(out[0]).save(f'{img_dir}/{prompt}.png')
                    
        # save checkpoint
        if j % 500 == 0: torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_latest.pth'))
        if j % 1000 == 0: torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
    
    evaluate(pretrain_detector, seg_module, (model, sampler, state), class_train, class_test, args.exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--class_split",
        type=int,
        help="the class split: 1,2,3,4,5,6",
        default=1
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="path to where to save training things",
        required=True
    )

    args = parser.parse_args()
    main(args)
