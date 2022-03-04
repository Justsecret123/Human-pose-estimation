# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:24:14 2022

@author: Ibrah
"""

import argparse
import tensorflow as tf
import time 
import cv2

def run_inference(parser):
    
    print("\n\nRunning inference\n-----------------------")
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Get the args as a dict(key,value)
    variables = vars(args)
    
    # Get the model and its parameters
    interpreter, input_details, output_details  = load_model(variables["model"])
    
    # Load the gif and its parameters
    gif, frame_count, video_writer = load_gif(variables["source"])
        
    #print(f"Vars: {variables}\n")
    

def load_model(path):
    
    print("Loading the model...\n-----------------------")
    
    # Intialize the timer
    timer = time.time()
    
    # Load the interpreter (model)
    interpreter = tf.lite.Interpreter(model_path=path)
    
    #Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Calculate and display the elapsed time
    timer = round(time.time() - timer,2)
    print(f"Model loaded in: {timer}s\n")
    
    return interpreter, input_details, output_details

def load_gif(path):
    
    print("Loading the gif...\n-----------------------")
    
    # Intialize the timer
    timer = time.time()
    
    # Load the gif 
    gif = cv2.VideoCapture(path)
     
    # Get the parameters 
    frame_count = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = []
    
    # Calculate and display the elapsed time
    timer = round(time.time() - timer,2)
    print(f"Gif loaded in: {timer}s\n")
     
    return gif, frame_count, video_writer

def create_parser():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run inferences on a gif")

    #Add arguments 
    
    # Model path 
    parser.add_argument("-model", help="Model path")
    
    # Source 
    parser.add_argument("-source", help="Source gif")
    
    # Destination
    parser.add_argument("-output", help="Destination of the output gif")
    
    # Frame reate
    parser.add_argument("-fps", type=int, help="Frame rate")
    
    # Threshold
    parser.add_argument("-thres", type=float, help="Detection threshold")
    
    # Thickness
    parser.add_argument("-thickness", type=int, help="Line thickness")
    
    return parser


if __name__ == "__main__": 
    
    # Initialize the parser
    parser = create_parser()
    
    # Main loop
    run_inference(parser)