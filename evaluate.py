import os
import argparse
import random
import torchvision
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from scripts.demo.turbo import *
from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from pytorch_lightning import seed_everything
from mmdet.apis import init_detector, inference_detector
from utils import load_classes, get_rand, IoU
from seg_module import Segmodule

warnings.filterwarnings("ignore")


def evaluate(pretrain_detector, seg_module, diffusion_model, 
             class_seen, class_unseen, exp_dir:str, 
             eval_iter:int = 10, H:int = 512, W:int = 512):
    
    model, sampler, state = diffusion_model

    print('***********************   begin   **********************************')
    print(f"Start evaluate with {eval_iter} iterations per class.")
    
    with torch.no_grad():
        for v, classes in enumerate([class_seen, class_unseen]):
            iou = 0
            total_iter = eval_iter * len(classes)
            class_embedding_dic = {}
            
            for class_name in classes:
                class_embedding, uc = get_cond(model, H=args.H, W=args.W, prompt=class_name)
                class_embedding = class_embedding['crossattn'][:, 0, :].unsqueeze(1)
                class_embedding_dic[class_name] = class_embedding
                
            for _ in tqdm(range(total_iter), desc = 'seen' if v == 0 else 'unseen'):
                trainclass = classes[random.randint(0, len(classes)-1)]
                prompt = "a photograph of a " + trainclass
                
                # generate images
                seed = get_rand()
                out = sample(
                    model, sampler, H=args.H, W=args.W, seed=seed, 
                    prompt=prompt, filter=state.get("filter")
                )
                diffusion_features = copy.copy(get_feature_dic())
                
                # detector
                result = inference_detector(pretrain_detector, out[0], text_prompt=trainclass)
                flag = True # detect if mmdet fail to detect the object
                seg_result = result.pred_instances.masks
                if len(seg_result) > 0: flag = False
                if flag: continue # "pretrain detector fail to detect the object
                gt_seg = seg_result[0].unsqueeze(0).float() # 1, 512, 512
                
                # get class embedding
                class_embedding = class_embedding_dic[trainclass]
                
                # seg_module
                pred_seg = seg_module(diffusion_features, class_embedding)
                pred_seg = pred_seg.squeeze(0)
                iou += IoU(pred_seg, gt_seg)

            with open(f'{exp_dir}/ious_{v}.txt', "w") as f:
                f.write(str(iou/total_iter)+'\n')
          
          
def main(args):
    seed_everything(args.seed)

    class_train, class_test, class_coco = load_classes(args.class_split)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_file = 'src/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
    pretrain_detector = init_detector(config_file, checkpoint_file, device=device)
    
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

    seg_module = Segmodule().to(device)
    seg_module.load_state_dict(torch.load(args.grounding_ckpt, map_location="cpu"), strict=True)
           
           
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
    parser.add_argument(
        "--grounding_ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )

    args = parser.parse_args()
    main(args)