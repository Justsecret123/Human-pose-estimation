# Human-pose-estimation

A quick tutorial on multi-pose estimation with OpenCV, Tensorflow and MoveNet lightning.

> **Also available on [Kaggle](https://www.kaggle.com/ibrahimserouis99/human-pose-estimation-with-movenet)

# Motivation 

I found multipose estimation Notebooks and codes not so explicit or even understandable for pure beginners. As a result, the idea of writing my own tutorial naturally came to me. After some research and a bit of styling, code cleaning, presentation...I've finally made it public. 

# Model info 

- Model type : MoveNet 
- Pose estimation method : multipose, bottom-up
- Keypoints count : 17

# Results (example)
![Results](https://github.com/Justsecret123/Human-pose-estimation/blob/main/Test%20gifs/results.gif)

# How to use 

## Command line runner
- [Test script](/Scripts/movenet_inference.py)
- [Bat file for a sample test](/Scripts/test_inference.bat)
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
