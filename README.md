# Human-Flow-Detection Project

This is an human flow detection project that use **YOLOv8n** model as detector, **DeepSORT** as tracker, and **OSNet** with **Torcheid** as person Re-ID model.

## Installation

**Install the following python dependency**

```python
pip install -r requirements.txt
```

## Image Annotation

Run the command: "labelImg" in terminal to access the labelImg GUI. Please maximize the window size and switch the annotation type to YOLO from the navigation bar on left side.

You might encounter error while drawing the bound due to returning float value instead of integer, please just copy the traceback code and ask ChatGPT how to resolve this issue by edit canvas.py since just need to copy the right code and replace the original line.

**This process might have to repeat multiple times until canvas.py had been proper.**

## Training Datasets Structure

The training dataset structure should be as below:

```kotlin
datasets/
│── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── val/
│       ├── image3.jpg
│       ├── image4.jpg
│── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    ├── val/
        ├── image3.txt
        ├── image4.txt
```

**You have to modify the data.yaml under `model_training/` to ensure the data path is correct.**

## Structure of Human-Flow-Detection

This repository contains of two system engine with different architecture. The structure of engine files is as listed below:

```python
engines/
    engine.py # Application of multiple cameras tracking
    zone.py # Application of zone partition and zone tracking
```

Both engines required two arguments, camera_ID and camera_Index. **Camera_ID** is a camera identifier that could be set as whatever value. **Camera_Index** is the system camera code, it should strictly following the default setting in order for OpenCV to capture the camera.

You may edit the `parameter.xml` to adjust the inference threshold and feature extraction threshold.

This repository also contains of the model training codes in `train.py` under `model_training/`.


