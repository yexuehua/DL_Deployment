## How to deploy yolov5 to real time detection

## 1.Installation

```shell
conda create -n yolov5 python=3.8
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## 2.Dataset

Download [coco128](https://www.kaggle.com/ultralytics/coco128)

Move the dataset to the following structure

|-- yolov5

|    |-- ...

|-- datasets

|    |-- coco128

|    |    |-- images

|    |    |-- labels

|    |    |-- ...

## 3.Run the code

Adding the "mydeploy.py" to the root folder of yolov5 and then run the following code

```
python mydeploy.py
```

