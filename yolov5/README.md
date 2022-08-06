## How to deploy yolov5 to real time detection

## 1.Installation

```shell
conda create -n yolov5 python=3.8
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## 2.Run the code

```
python mydeploy.py
```

