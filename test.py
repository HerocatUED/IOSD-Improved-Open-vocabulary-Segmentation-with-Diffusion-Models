import argparse
import os
import torch
import torchvision
import numpy as np

from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from scripts.demo.turbo import *
from utils import chunk, plot_mask, get_rand
from seg_module import Segmodule


def main(args):

    seed_everything(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
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

    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir
    batch_size = args.n_samples

    assert batch_size == 1 
    
    with torch.no_grad():
        prompt = args.prompt
        trainclass = args.category
        
        if not args.from_file:
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {args.from_file}")
            with open(args.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        for prompts in data:

            # generate images
            seed = get_rand()
            out = sample(
                model, sampler, H=args.H, W=args.W, seed=seed, 
                prompt=prompts[0], filter=state.get("filter")
            )
            diffusion_features = copy.copy(get_feature_dic())
            img = out[0]
            Image.fromarray(img).save(f"{sample_path}/{args.prompt}.png")

            # get class embedding
            class_embedding, uc = sample(
                model, sampler, condition_only=True, H=args.H, W=args.W, seed=seed, 
                prompt=trainclass, filter=state.get("filter")
            )
            class_embedding = class_embedding['crossattn'][:, 0, :].unsqueeze(1)

            # seg_module
            total_pred_seg = seg_module(diffusion_features, class_embedding)

            pred_seg = total_pred_seg[0]
            label_pred_prob = torch.sigmoid(pred_seg)
            label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
            label_pred_mask[label_pred_prob > 0.5] = 1
            annotation_pred = label_pred_mask.cpu()

            mask = annotation_pred.numpy()
            mask = np.expand_dims(mask, 0)
            done_image_mask = plot_mask(img, mask, alpha=0.9, indexlist=[0])
            
            print(f"saving to {sample_path}")
            Image.fromarray(done_image_mask.reshape((512, 512, 3))).save(os.path.join(f"{sample_path}/{args.prompt}_mask.png"))

            torchvision.utils.save_image(annotation_pred, os.path.join(
                f"{sample_path}/{args.prompt}_seg.png"), normalize=True, scale_each=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="the prompt to render"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="?",
        default="lion",
        help="the category to ground"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="number of sampling steps",
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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--grounding_ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    args = parser.parse_args()
    main(args)
