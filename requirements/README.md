# Installation
0. Before set up environment, make sure all submodule have been cloned and its requirements have been installed, if not, run
```bash
git submodule init && git submodule update
cd src/mmdetection && pip install -v -e .
cd src/taming-transformers && pip install -v -e .
```

1. First we recommend to create a new conda environment and activate it, run
```bash
conda create --name mml python=3.8
conda activate mml
```

2. The project requires one extra directory, run
```bash
mkdir checkpoint
```

3. You need to install basic requirements, run
```bash
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -r requirements.txt
pip install -r pt2.txt
```

4. Last, we nee to copy config file, run
```bash
cp -rp configs/mmdetection/swin/ src/mmdetection/configs/
```

