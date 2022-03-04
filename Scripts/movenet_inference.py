# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:24:14 2022

@author: Ibrah
"""

import argparse
import tensorflow as tf
import time 

def run_inference(parser):
    
    print("Running inference : ")
    args = parser.parse_args()
    

def load_model():
    
    print("\n\nLoading the model...\n\n")
    
    # Intialize the timer
    timer = time.time()
    
    # Load the interpreter (model)
    interpreter = tf.lite.Interpreter(model_path="../Model/TFLite/lite-model_movenet_multipose_lightning_tflite_float16_1.tflite")
    
    #Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Calculate and display the elapsed time
    timer = round(time.time() - timer,2)
    print(f"\n\nElapsed time: {timer}s\n\n")
    
    return interpreter, input_details, output_details


def create_parser():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run inferences on a gif")

    #Add arguments 
    
    # Source 
    parser.add_argument("-source", type=ascii, help="Source gif")
    
    # Destination
    parser.add_argument("-output", type=ascii, help="Destination of the output gif")
    
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
    
    # Load the model
    movenet, input_details, output_details = load_model()
    
    # Detect 
    
    
    run_inference(parser)