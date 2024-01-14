# Multimodal-Learning
Project of Multimodal Learning (PKU 2023 Autumn)

This project is based on [Grounded-Diffusion](https://github.com/Lipurple/Grounded-Diffusion), 
but we modified the whole code base because the original codes are too ugly and hard to use.
We only remained the idea and reconstructed the whole project based on official stable diffusion code base, 
so as to easy extension.

TODO List：
- [x] Reproduce
- [x] Clean the code base
- [x] Try up-to-date stable diffusion models(we modified the whole code base)
- [x] Explorey frozen word embeddings
- [x] Modify fusion module with advanced techniques
- [ ] Check training classes, remove those mmdet can't segment
- [ ] Build prompts, not always "a photography of ..." 
- [ ] Batch Size Problem
- [x] lr schedule
- [ ] Explore uc to seg_module
- [x] Optional: web UI inference demo
- [ ] Optional: Try segment a given image rather than segment generated images. 
- [ ] Optional: Try Stable Video Diffusion via video segmentation
- [ ] Experiments

## Requirements
1. Install [pytorch](https://pytorch.org/) (we use 2.1.1 with cuda 11.8)
2. Install requirements, see instructions under requirements folder
3. Make sure you have access to hugging face (If not, just put ```HF_ENDPOINT=https://hf-mirror.com``` before all commands bellow)

## Model Zoo
Put these models under `checkpoint` folder:
1. [diffusion model(sd_xl_turbo_1.0_fp16.safetensors)](https://huggingface.co/stabilityai/sdxl-turbo/tree/main)
2. [detection model]()
3. [segmentation module]()

## Demo
After you have your seg_model, you can run a real-time web UI with following command:
```streamlit run demo.py```

## Train & Inference
Before training, please download the diffusion model and detection model into a folder called `checkpoint`. 

See *command.txt*
	
## Acknowledgements
Many thanks to the code bases from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [CLIP](https://github.com/openai/CLIP), [taming-transformers](https://github.com/CompVis/taming-transformers), [mmdetection](https://github.com/open-mmlab/mmdetection)
, [Stablility-AI:generative-models](https://github.com/Stability-AI/generative-models)