import argparse
import cv2
import numpy as np
import os

from inference import Network


image_type = ['cardboard','glass','metal','paper','plastic','trash']

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
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)
    image_name = os.path.basename(args.i).split()[0]
    ### TODO: Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()
    result = output['dense_2/Softmax']
    pred = np.argmax(result,axis = 1)
    pred_value = np.max(pred)
    output_image = create_output_image(image, pred_value)
    cv2.imwrite("output/{}.jpg".format(image_name),output_image)
    

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
    perform_inference(args)


if __name__ == "__main__":
    main()
