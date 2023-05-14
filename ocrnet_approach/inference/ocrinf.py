import inference
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(
    prog='ocrinf',
    description='inference script for ocr approach'
)

parser.add_argument('onnx_model_file', help='Filename of onnx model')
parser.add_argument('input_image_file', help='Filename of an input image')
parser.add_argument('output_file', help='Filename where to save the result')

parser.add_argument('-m', '--multiplied', action='store_true', help='Return multiplied image')
parser.add_argument('-s', '--size', choices=['256', '500'], default='500',
                    help='Size of an image that the model operates on')
parser.add_argument('-c', '--colors', choices=['rgb', 'coffee'], default='rgb',
                    help='color palette to use for segmentation')


def param2color(parameter: str) -> np.ndarray:
    colors = None
    if parameter == 'rgb':
        colors = np.array([[255, 255, 255],  # Background
                           [255, 128, 128],  # Monolayer
                           [128, 255, 128],  # Bilayer
                           [128, 128, 255]])  # Three layer
    elif parameter == 'coffee':
        colors = np.array([[227, 217, 207],  # Background
                           [176, 179, 162],  # Monolayer
                           [109, 121, 117],  # Bilayer
                           [63, 56, 46]])  # Three layer
    return colors


def param2size(parameter: str) -> tuple:
    return int(parameter), int(parameter)


if __name__ == '__main__':
    args = parser.parse_args()

    # define filenames
    picture_filename = args.input_image_file
    onnx_filename = args.onnx_model_file
    output_filename = args.output_file

    # load the model
    onnx_model = inference.load_model(onnx_filename)

    # get segmentations
    segm = inference.get_segmentation(onnx_model, picture_filename,
                                      size=param2size(args.size),
                                      colors=param2color(args.colors),
                                      multiplied=args.multiplied)

    # save results
    Image.fromarray(segm).save(output_filename)
