# Human-pose-estimation ![Language_support](https://img.shields.io/pypi/pyversions/Tensorflow) ![Last_commit](https://img.shields.io/github/last-commit/JustSecret123/Human-pose-estimation) ![Workflow](https://img.shields.io/github/workflow/status/JustSecret123/Human-pose-estimation/Pylint/main) ![Tensorflow_version](https://img.shields.io/badge/Tensorflow%20version-2.6.2-orange) 

A quick tutorial on multi-pose estimation with OpenCV, Tensorflow and MoveNet lightning.

> **Also available on [Kaggle](https://www.kaggle.com/ibrahimserouis99/human-pose-estimation-with-movenet) 
<a href="https://www.linkedin.com/in/ibrahim-serouis-b05378181/">
  <img src="https://img.shields.io/badge/LinkedIn-Ibrahim%20Serouis-blue?link=http://left&link=http://right)"/>
</a>


# Motivation 

I found multipose estimation Notebooks and codes not so explicit or even understandable for pure beginners. Moreover, most of the available tutorials focus on single-pose estimation, with only one instance (human).  As a result, the idea of writing my own tutorial naturally came to me. After some research and a bit of styling, code cleaning, presentation...I finally made it public. 

# Model info 

- Model type : MoveNet 
- Pose estimation method : multipose, bottom-up
- Keypoint count : 17

# Results (example)
![Results](https://github.com/Justsecret123/Human-pose-estimation/blob/main/Screenshots/results.gif)

# How to use 

## Command line runner
- [Test script](/Scripts/movenet_inference.py)
- [Bat file for a sample test](/Scripts/test_inference.bat) : requires the model path to be *../Model/TFLite/lite-model_movenet_multipose_lightning_tflite_float16_1.tflite*. You can [download it directly from TFHub](https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1)
> Args :
![Command_line_args](/Screenshots/command_line_args.PNG)



## Notebook 

- [Kaggle](https://www.kaggle.com/ibrahimserouis99/human-pose-estimation-with-movenet)
> Requires Tensorflow Hub and a Kaggle environment. However, feel free to adapt to Notebook to your local setup


# Required packages

- Deep Learning and calculations : Tensorflow 2.x, NumPy, Tensorflow Docs
- Computer graphics/vision : OpenCV 
- Display : IPython, Matplotlib 
- Image/video writer : image io

# Acknowledgements 
- Tensorflow official tutorial 
