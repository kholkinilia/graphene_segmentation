import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2


def _load_image(image_filename: str) -> np.ndarray:
    """
    :param image_filename: filename of an image

    Load an image as a numpy array from the filename
    """
    return np.array(Image.open(image_filename)).astype(np.float32) / 255.


def _resize(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param size: tuple of length 2 representing the height and width of the picture

    Resize an image to a needed size
    """
    return cv2.resize(image, size)


def _normalize(image: np.ndarray) -> np.ndarray:
    """
    :param image: image represented as np.ndarray

    Normalize an image as if it's from ImageNet
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    image -= IMAGENET_MEAN[None, None, :]
    image /= IMAGENET_STD[None, None, :]
    return image


def _median_correction(image: np.ndarray, intensity=0.6) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param intensity: intensity of a median pixel in the resulting image

    Apply median correction
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rmedian, gmedian, bmedian = np.median(r), np.median(g), np.median(b)
    result = image
    result[:, :, 0] *= intensity / rmedian
    result[:, :, 1] *= intensity / gmedian
    result[:, :, 2] *= intensity / bmedian
    return np.clip(result, 0., 1.)


def _prepare_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param size: a tuple representing the size of the resulting picture

    Prepare an image for fitting to the model
    """
    return _normalize(_resize(_median_correction(image), size)).transpose((2, 0, 1))[None, :, :, :]


def _get_pred(model: ort.InferenceSession, image: np.ndarray, input_name: str = 'input') -> np.ndarray:
    """
    :param model: onnx inference session
    :param image: image represented as np.ndarray
    :param input_name: the name of the input of onnx model (default: 'input')

    Get a prediction for an image
    """
    return model.run(None, {input_name: image})[1]


def _pred2segm(pred: np.ndarray, classes_ids: np.ndarray = np.arange(4), colors: np.ndarray = None) -> np.ndarray:
    """
    :param pred: prediction got from onnx inference session
    :param classes_ids: indices of classes
    :param colors: colors corresponding to classes. np.ndarray of a shape (n_classes, 3), where the last dimension specifies color of a class

    Transfers model prediction to segmentation picture
    """
    if colors is None:
        colors = np.array([[227, 217, 207],  # Background
                           [176, 179, 162],  # Monolayer
                           [109, 121, 117],  # Bilayer
                           [63, 56, 46]])  # Three layer
        # colors = np.array([[255, 255, 255], # Background
        #                    [255, 128, 128], # Monolayer
        #                    [128, 255, 128], # Bilayer
        #                    [128, 128, 255]])   # Three layer

    # create segmentation from predictions
    pred = pred.argmax(axis=1, keepdims=True)
    segm = np.zeros((pred.shape[0], 3, pred.shape[2], pred.shape[3]))
    for ind in classes_ids:
        segm += (pred == ind) * colors[ind][None, :, None, None]

    return segm.transpose((0, 2, 3, 1))


def load_model(model_filename: str) -> ort.InferenceSession:
    """
    :param model_filename: filename of onnx model

    Create an onnx inference session from the filename
    """
    return ort.InferenceSession(model_filename)


def get_segmentation(model: ort.InferenceSession, image_filename: str, size: tuple = (256, 256),
                     colors: np.ndarray = None) -> np.ndarray:
    """
    :param model: onnx inference session
    :param image_filename: a filename for an image
    :param size: a size of the picture that model accepts as an input
    :param colors: colors of classes. np.ndarray for a shape (n_classes, 3), where the last dimension specifies color

    Get segmentation from an onnx inference session and an image filename
    """
    image = _load_image(image_filename)
    initial_size = (image.shape[1], image.shape[0])
    image = _prepare_image(image, size)
    pred = _get_pred(model, image)
    segm = _pred2segm(pred, colors=colors).squeeze()
    resized_segm = cv2.resize(segm, initial_size)
    return resized_segm.astype(np.uint8)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    picture_filename = '../resources/dataset/train/pure/raw/2-1-100x mono.jpg'
    onnx_filename = '../resources/models/OCRNet/onnx/ocrnet_resnet256.onnx'
    onnx_model = load_model(onnx_filename)

    segm = get_segmentation(onnx_model, picture_filename)

    plt.imshow(segm)
    plt.axis('off')
