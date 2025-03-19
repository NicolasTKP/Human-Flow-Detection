# Human-Flow-Detection Project

This is an human flow detection project that use **YOLOv8n** Model

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

**You have to modify the data.yaml to ensure the data path is correct.**
