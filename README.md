Local install
```
python -m venv venv
source venv/Scripts/activate
```
```
git clone https://github.com/antonstrobe/automatic-number-plate-recognition-python-yolov8
cd automatic-number-plate-recognition-python-yolov8
pip install -r requirements.txt
```
```
pip install torch torchvision
pip install opencv-python
pip install pytesseract
pip install ultralytics
git clone https://github.com/antonstrobe/sort
cd sort
pip install -r requirements.txt
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install pytorch-lightning
```
```
cd ..
python main.py
```



Узнать какие версии питона есть
```
%USERPROFILE% \AppData\Local\Programs\Python\
```


# automatic-number-plate-recognition-python-yolov8

<p align="center">
<a href="https://www.youtube.com/watch?v=fyJB1t0o0ms">
</a>
</p>

## data

The video I used in this tutorial can be downloaded [here](https://drive.google.com/file/d/12sBfgLICdQEnDSOkVFZiJuUE6d3BeanT/view?usp=sharing).

## models

A Yolov8 pretrained model was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and following this [step by step tutorial on how to train an object detector with Yolov8 on your custom data](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide). 

The trained model is available in my [Patreon](https://www.patreon.com/ComputerVisionEngineer).

## dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort) as mentioned in the [video](https://youtu.be/fyJB1t0o0ms?t=1120).
