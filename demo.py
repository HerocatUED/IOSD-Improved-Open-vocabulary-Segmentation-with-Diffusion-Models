import argparse
import torch
import numpy as np

from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from scripts.demo.turbo import *
from utils import plot_mask
from seg_module import Segmodule


def demo(args):
    st.title("Turbo Segmentation")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seg_module = Segmodule().to(device)
    seg_module.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    
    head_cols = st.columns([1, 1, 1])
    with head_cols[0]:
        version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
        version_dict = VERSION2SPECS[version]

    with head_cols[1]:
        v_spacer(2)
        if st.checkbox("Load Model"):
            mode = "txt2img"
        else:
            mode = "skip"

    if mode != "skip":
        state = init_st(version_dict, load_filter=True)
        if state["msg"]:
            st.info(state["msg"])
        model = state["model"]
        load_model(model)

    # seed
    if "seed" not in st.session_state:
        st.session_state.seed = 0

    def increment_counter():
        st.session_state.seed += 1

    def decrement_counter():
        if st.session_state.seed > 0:
            st.session_state.seed -= 1

    with head_cols[2]:
        n_steps = st.number_input(label="number of steps", min_value=1, max_value=4)
    
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    sampler.n_sample_steps = n_steps
    default_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    default_catogory = "priest robe"
    prompt = st_keyup("Enter a value", value=default_prompt, debounce=300, key="interactive_text")
    catogory = st_keyup("Enter a value", value=default_catogory, debounce=300, key="interactive_text")
    

    cols = st.columns([1, 5, 1])
    if mode != "skip":
        with cols[0]:
            v_spacer(14)
            st.button("↩", on_click=decrement_counter)
        with cols[2]:
            v_spacer(14)
            st.button("↪", on_click=increment_counter)

        sampler.noise_sampler = SeededNoise(seed=st.session_state.seed)
        out = sample(
            model, sampler, H=512, W=512, seed=st.session_state.seed, prompt=prompt, filter=state.get("filter")
        )
        img = out[0]
        # get class embedding
        class_embedding, uc = sample(
            model, sampler, condition_only=True, H=512, W=512, seed=st.session_state.seed, 
            prompt=catogory, filter=state.get("filter")
        )
        class_embedding = class_embedding['crossattn']
        if class_embedding.size()[1] > 1:
            class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
        class_embedding = class_embedding.repeat(1, 1, 1)

        # seg_module
        diffusion_features = get_feature_dic()
        total_pred_seg = seg_module(diffusion_features, class_embedding)

        pred_seg = total_pred_seg[0]
        label_pred_prob = torch.sigmoid(pred_seg)
        label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
        label_pred_mask[label_pred_prob > 0.5] = 1
        annotation_pred = label_pred_mask.cpu()

        mask = annotation_pred.numpy()
        mask = np.expand_dims(mask, 0)
        done_image_mask = plot_mask(img, mask, alpha=0.6, indexlist=[0]).reshape((512, 512, 3))
        output = np.concatenate(img.cpu().numpy(), done_image_mask)
        with cols[1]:
            st.image(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    
    args = parser.parse_args()
    demo(args)
