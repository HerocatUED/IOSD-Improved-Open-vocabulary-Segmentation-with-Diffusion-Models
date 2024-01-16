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
from scripts.demo.turbo import *
from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from pytorch_lightning import seed_everything
from mmdet.apis import init_detector, inference_detector
from utils import chunk, get_rand
from seg_module import Segmodule
from evaluate import evaluate

warnings.filterwarnings("ignore")


def load_classes(args):
    print("Loading classes from COCO and PASCAL")
    class_coco = {}
    f = open("configs/data/coco_80_class.txt", "r")
    count = 0
    for line in f.readlines():
        c_name = line.split("\n")[0]
        class_coco[c_name] = count
        count += 1

    pascal_file = f"configs/data/VOC/class_split{args.class_split}.csv"
    class_total = []
    f = open(pascal_file, "r")
    count = 0
    for line in f.readlines():
        count += 1
        class_total.append(line.split(",")[0])
    class_train = class_total[:15]
    class_test = class_total[15:]
    
    return class_train, class_test, class_coco


def main(args):
    
    seed_everything(args.seed)

    class_train, class_test, class_coco = load_classes(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_file = 'src/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
    pretrain_detector = init_detector(config_file, checkpoint_file, device=device)
    
    seg_module = Segmodule().to(device)

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
    
    learning_rate = 1e-4
    total_iter = 50000
    g_optim = optim.Adam(
        [{"params": seg_module.parameters()},],
        lr=learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=15000, gamma=0.5)

    print('***********************   begin   **********************************')
    print(f"Start training with maximum {total_iter} iterations.")


    for j in range(1, total_iter+1):
        lr_scheduler.step()
        print('Iter ' + str(j) + '/' + str(total_iter))
        if not args.from_file:
            trainclass = class_train[random.randint(0, len(class_train)-1)]
            prompt = "a photograph of a " + trainclass
            print(f"Iter {j}: prompt--{prompt}")
        else:
            raise NotImplementedError
            print(f"reading prompts from {args.from_file}")
            with open(args.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
        # class_index = class_coco[trainclass]
        
        # generate images
        seed = get_rand()
        out = sample(
            model, sampler, H=args.H, W=args.W, seed=seed, 
            prompt=prompt, filter=state.get("filter")
        )
        x_sample_list = out[0]
        diffusion_features = copy.copy(get_feature_dic())
        
        # detector
        result = inference_detector(pretrain_detector, x_sample_list, text_prompt=trainclass)
        flag = True # detect if mmdet fail to detect the object
        seg_result = result.pred_instances.masks[0].unsqueeze(0)
        if len(seg_result) > 0: flag = False
        if flag: continue # "pretrain detector fail to detect the object
        
        # get class embedding
        class_embedding, uc = sample(
            model, sampler, condition_only=True, H=args.H, W=args.W, seed=seed, 
            prompt=trainclass, filter=state.get("filter")
        )
        class_embedding = class_embedding['crossattn']
        if class_embedding.size()[1] > 1:
            class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
        class_embedding = class_embedding.repeat(1, 1, 1)
        
        # seg_module
        pred_seg = seg_module(diffusion_features, class_embedding)
        pred_seg = pred_seg.squeeze(0)
        gt_seg = seg_result.float() # 1, 512, 512
        loss = F.binary_cross_entropy_with_logits(pred_seg, gt_seg)
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        writer.add_scalar('train/loss', loss.item(), global_step=j)
        
        # visualization 
        if  j % 200 == 0:
            pred_seg = torch.sigmoid(pred_seg)
            pred_seg[pred_seg <= 0.5] = 0
            pred_seg[pred_seg > 0.5] = 1
            viz = torch.cat([gt_seg, pred_seg], axis=1)
            torchvision.utils.save_image(viz, 
                            img_dir +'/viz_sample_{0:05d}_seg'.format(j)+trainclass+'.png', 
                            normalize=True, scale_each=True)
            Image.fromarray(out[0]).save(f'{img_dir}/{prompt}.png')
                    
        # save checkpoint
        if j % 200 == 0: torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_latest.pth'))
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
        help="the class split: 1,2,3",
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
