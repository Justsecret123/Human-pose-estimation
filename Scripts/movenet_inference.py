# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:24:14 2022

@author: Ibrah
"""

import argparse
import time
import cv2
import imageio
import numpy as np
import tensorflow as tf


# Dimensions
WIDTH = HEIGHT = 256

# Colors : RGB content
cyan = (255, 255, 0)
magenta = (255, 0, 255)

# Edge colors
EDGE_COLORS = {
    (0, 1): magenta,
    (0, 2): cyan,
    (1, 3): magenta,
    (2, 4): cyan,
    (0, 5): magenta,
    (0, 6): cyan,
    (5, 7): magenta,
    (7, 9): cyan,
    (6, 8): magenta,
    (8, 10): cyan,
    (5, 6): magenta,
    (5, 11): cyan,
    (6, 12): magenta,
    (11, 12): cyan,
    (11, 13): magenta,
    (13, 15): cyan,
    (12, 14): magenta,
    (14, 16): cyan,
}


def main(main_parser):
    """Main loop"""
    
    # Parse the command line arguments
    args = main_parser.parse_args()

    # Get the args as a dict(key,value)
    variables = vars(args)

    # Get the model and its parameters
    interpreter, input_details, output_details = load_model(variables["model"])

    # Load the gif and its parameters
    gif, video_writer, duration, initial_shape = load_gif(variables["source"])

    print(f"Vars:\n-----------------------\n{variables}")

    threshold = variables["thres"]
    frame_rate = variables["fps"]
    thickness = variables["thickness"]
    destination = variables["output"]

    # Initialize the timer
    timer = time.time()

    # Perform the inference
    print("\n\nRunning inference\n-----------------------")
    video_writer = inference(
        gif,
        interpreter,
        input_details,
        output_details,
        video_writer,
        thickness,
        threshold,
        initial_shape,
    )

    # Calculate and display the elapsed time
    timer = round(time.time() - timer, 2)
    print(f"(Inference drawing) total time: {timer}s for a {duration}s video.\n")

    # Save the results
    save_results(video_writer, frame_rate, destination)


def inference(
    gif,
    interpreter,
    input_details,
    output_details,
    output_frames,
    thickness,
    threshold,
    initial_shape
):
    """Runs inferences on each frame"""
    
    while gif.isOpened():

        # Capture the frame
        ret, frame = gif.read()

        # Process the frame : resize to the input size
        if frame is None:
            break

        # Copy the frame
        image = frame.copy()
        image = cv2.resize(image, (WIDTH, HEIGHT))

        # Create a batch (input tensor)
        input_tensor = tf.expand_dims(image, axis=0)
        input_tensor = tf.cast(input_tensor, dtype=tf.uint8)

        # Setup
        is_dynamic_shape_model = input_details[0]["shape_signature"][2] == -1
        if is_dynamic_shape_model:
            input_tensor_index = input_details[0]["index"]
            input_shape = input_tensor.shape
            interpreter.resize_tensor_input(
                input_tensor_index, input_shape, strict=True
            )

        # Set the input tensor and invoke the interpreter
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], input_tensor.numpy())
        interpreter.invoke()

        # Perform inference
        results = interpreter.get_tensor(output_details[0]["index"])

        """
        Output shape : [1, 6, 56] ---> (batch size), (instances), (xy keypoints coordinates and score from [0:50] 
        and [ymin, xmin, ymax, xmax, score]
        for the remaining elements)
        First, let's resize it to a more convenient shape, following this logic : 
        - First channel ---> each instance
        - Second channel ---> 17 keypoints for each instance
        - The 51st values of the last channel ----> the confidence score.
        Thus, the Tensor is reshaped without losing important information.    
        """

        keypoints = results[:, :, :51].reshape((6, 17, 3))

        # Loop through the results
        loop(image, keypoints, threshold, thickness)

        # Get the output frame : reshape to the original size
        frame_rgb = cv2.cvtColor(
            cv2.resize(
                image,
                (initial_shape[0], initial_shape[1]),
                interpolation=cv2.INTER_LANCZOS4,
            ),
            cv2.COLOR_BGR2RGB,
        )  # OpenCV processes BGR images instead of RGB

        # Add the drawings to the output frames
        output_frames.append(frame_rgb)

    # Release the object
    gif.release()

    return output_frames


def loop(frame, keypoints, threshold, thickness):
    """Loops through the inference results for each human,
        then proceeds to draw the associated keypoints and edges
    """
    
    # Loop through the results
    for instance in keypoints:
        # Draw the keypoints and get the denormalized coordinates
        denormalized_coordinates = draw_keypoints(frame, instance, threshold)
        # Draw the edges
        draw_edges(
            denormalized_coordinates, frame, EDGE_COLORS, threshold, thickness
        )


def load_model(path):
    """Loads the TFLite model"""

    print("Loading the model...\n-----------------------")

    # Intialize the timer
    timer = time.time()

    # Load the interpreter (model)
    interpreter = tf.lite.Interpreter(model_path=path)

    # Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Calculate and display the elapsed time
    timer = round(time.time() - timer, 2)
    print(f"Model loaded in: {timer}s\n")

    return interpreter, input_details, output_details


def load_gif(path):
    """Loads the gif and returns its parameters"""

    print("Loading the gif...\n-----------------------")

    # Intialize the timer
    timer = time.time()

    # Load the gif
    gif = cv2.VideoCapture(path)

    # Initialize the video writer
    video_writer = []

    # Get the parameters
    fps = gif.get(cv2.CAP_PROP_FPS)
    frame_count = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # Get the initial shape
    initial_shape = []
    initial_shape.append(int(gif.get(cv2.CAP_PROP_FRAME_WIDTH)))
    initial_shape.append(int(gif.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Calculate and display the elapsed time
    timer = round(time.time() - timer, 2)
    print(f"Gif loaded in: {timer}s\n")

    return gif, video_writer, duration, initial_shape


def draw_keypoints(frame, keypoints, threshold):
    """Draws the keypoints on a frame"""

    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    denormalized_coordinates = np.squeeze(np.multiply(keypoints, [WIDTH, HEIGHT, 1]))

    # Iterate
    for keypoint in denormalized_coordinates:

        # Unpack the keypoint values : y, x, confidence score
        keypoint_y, keypoint_x, keypoint_confidence = keypoint

        if keypoint_confidence > threshold:
            """
            Draw the circle
            Note : A thickness of -1 px will fill the circle shape by the specified color.
            """
            cv2.circle(
                img=frame,
                center=(int(keypoint_x), int(keypoint_y)),
                radius=4,
                color=(255, 0, 0),
                thickness=-1,
            )

    return denormalized_coordinates


def draw_edges(denormalized_coordinates, frame, edges_colors, threshold, thickness):
    """Draws the edges on a frame"""

    # Iterate through
    for edge, color in edges_colors.items():

        # Get the dict value associated to the actual edge
        line_start, line_end = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[line_start]
        y2, x2, confidence_2 = denormalized_coordinates[line_end]

        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) and (confidence_2 > threshold):
            cv2.line(
                img=frame,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,  # Gives anti-aliased (smoothed) line which looks great for curves
            )


def save_results(output_frames, frame_rate, destination):
    """Converts the output stack to a gif"""

    print("\nSaving the results...\n-----------------------")

    # Stack the output frames to compose a sequence
    output = np.stack(output_frames, axis=0)
    # Write the sequence to a gif
    imageio.mimsave(destination, output, fps=frame_rate)

    print(f"Results saved at : {destination}")


def create_parser():
    """Creates a parser for the command line runner"""
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run inferences on a gif")

    # Add arguments

    # Model path
    parser.add_argument("-model", help="Model path")

    # Source
    parser.add_argument("-source", help="Source gif")

    # Destination
    parser.add_argument("-output", default="result.gif", help="Destination of the output gif")

    # Frame reate
    parser.add_argument("-fps", type=int, help="Frame rate")

    # Threshold
    parser.add_argument("-thres", type=float, default=0.11, help="Detection threshold")

    # Thickness
    parser.add_argument("-thickness", type=int, help="Line thickness")

    return parser


if __name__ == "__main__":

    # Initialize the parser
    parser = create_parser()

    # Main loop
    main(parser)
