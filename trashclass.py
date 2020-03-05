# Trash classification app

import argparse
import cv2
import numpy as np
import socket
import json
from random import randint
from inference2 import Network
###Import any libraries for MQTT and FFmpeg
import paho.mqtt.client as mqtt
import sys

MQTT_HOST = '52.90.89.176'  # IPADDRESS
MQTT_PORT = 1883  # 3001 # Set the Port for MQTT
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

from inference2 import Network

image_type = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)

    return image


def infer_on_video(args):
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Create a Network for using the Inference Engine
    inference_network = Network()

    # Load the network model into the IE
    n, c, h, w = inference_network.load_model(args.m, args.d, CPU_EXTENSION)
    # net_input_shape = inference_network.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        sys.stdout.flush()
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        preprocessed_frame = preprocessing(frame, h, w)

        # Perform inference on the frame
        inference_network.sync_inference(preprocessed_frame)

        # Obtain the output of the inference request
        output = inference_network.extract_output()
        result = output['dense_2/Softmax']
        pred = np.max(np.argmax(result, axis=1))

        # print(pred)  
        # Send clases
        MQTT_MSG_SPEED = json.dumps({"class": image_type[pred]})
        client.publish(topic="class", payload=MQTT_MSG_SPEED, qos=0, retain=False)

        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ###Disconnect from MQTT
    client.disconnect()


def create_output_image(image, output):
    img_text = image_type[output]
    print(img_text)
    scaler = max(int(image.shape[0] / 1000), 1)
    image = cv2.putText(image, img_text,
                        (50 * scaler, 210 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                        scaler * 2, (5, 5, 250), 3 * scaler)

    return image


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()