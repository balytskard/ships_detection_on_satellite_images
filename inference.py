from train import *

import os
import re
import pandas as pd
import numpy as np
import cv2
import tqdm
from skimage.morphology import label


def validate_pixel_string(pixel_string):
    """Checks if a pixel string is valid.

    Args:
        pixel_string: A string of space-separated integers representing pixel positions and run lengths.

    Returns:
        A fixed string if the input is valid, otherwise None.
    """
    pixels = [int(x) for x in pixel_string.split()]

    pairs = list(zip(pixels[::2], pixels[1::2]))
    lists = [[pix for pix in range(pair[0], pair[0] + pair[1])] for pair in pairs]

    merged_lists = []
    while lists:
        current_list = lists.pop(0)
        merged = False
        for i in range(len(lists) - 1, -1, -1):
            other_list = lists[i]
            if set(current_list) & set(other_list):
                merged_list = list(set(current_list) | set(other_list))
                current_list = merged_list
                lists.pop(i)
                merged = True
        merged_lists.append(current_list)

    new_pairs = [[lst[0], len(lst)] for lst in merged_lists]
    sort = sorted(new_pairs, key=lambda x: x[0])

    result = []
    for sublist in sort:
        result.extend(sublist)

    return ' '.join(map(str, result))


def rle_encode(mask):
    """Encodes a binary mask into run-length encoded format, handling non-square shapes.

    Args:
        mask: A 2D numpy array representing the binary mask.

    Returns:
        A string representing the run-length encoded format of the mask.
    """
    flat_mask = mask.flatten()
    rows, cols = mask.shape

    starts = np.where(np.diff(np.concatenate(([0], flat_mask, [0])))[1:] > 0)[0] + 1
    runs = np.where(np.diff(np.concatenate(([0], flat_mask, [0])))[1:] < 0)[0]

    encoded_pixels = []
    for start, run in zip(starts, runs):
        row = start // cols
        col = start % cols
        encoded_pixels.append((col + 1, row + 1, run + 1))

    return ' '.join(map(str, encoded_pixels))


def predict_and_encode(model, image_path):
    """Predicts the mask for an image and encodes it in run-length format.

    Args:
        model: The trained segmentation model.
        image_path: The path to the image file.

    Returns:
        A tuple containing the image ID and the encoded mask.
        If no ship in image is predicted, returns image_id and empty string.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)

    if np.max(mask) == 0:
        return os.path.basename(image_path), ''

    labels = label(mask)

    encoded_pixels = []
    for i in range(labels.max()+1):
        if encoded_pixels:
            pixels = " ".join(re.findall(r'\d+', encoded_pixels[0]))
            result = validate_pixel_string(pixels)
            return os.path.basename(image_path), result
        else:
            return os.path.basename(image_path), ''


def create_submission(test_dir, output_file, model):
    """Creates a submission CSV file with predictions for images in a given directory.

    Args:
        test_dir (str): Path to the directory containing test images.
        output_file (str): Name of the output CSV file.

    Returns:
        None.
    """
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

    data = {'ImageId': [],
            'EncodedPixels': []}

    with tqdm.tqdm(total=len(image_paths), desc='Creating Predictions') as pbar:
        for image_path in image_paths:
            image_id, encoded_pixels = predict_and_encode(model, image_path)
            data['ImageId'].append(image_id)
            data['EncodedPixels'].append(encoded_pixels)
            pbar.update()

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    model = build_model(input_shape=(256, 256, 3))
    model.load_weights('model.h5')
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=dice_coef_loss,
                  metrics=[dice_coef, iou_coef])

    test = 'airbus-ship-detection/test_v2'
    output_file = 'submission.csv'
    model_path = 'model.h5'

    model.load_weights(model_path)
    create_submission(test, output_file, model)
