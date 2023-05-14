import inference
import argparse
from PIL import Image

parser = argparse.ArgumentParser(
    prog='ocrinf',
    description='inference script for ocr approach'
)

parser.add_argument('onnx_model_file', help='Filename of onnx model')
parser.add_argument('input_image_file', help='Filename of an input image')
parser.add_argument('output_file', help='Filename where to save the result')

parser.add_argument('-m', '--multiplied', action='store_true', help='Return multiplied image')

if __name__ == '__main__':
    args = parser.parse_args()

    # define filenames
    picture_filename = args.input_image_file
    onnx_filename = args.onnx_model_file
    output_filename = args.output_file

    # load the model
    onnx_model = inference.load_model(onnx_filename)

    # get segmentations
    segm = inference.get_segmentation(onnx_model, picture_filename, multiplied=args.multiplied)

    # save results
    Image.fromarray(segm).save(output_filename)
