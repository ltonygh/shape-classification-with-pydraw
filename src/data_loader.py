import os
import json
from typing import List, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import logging

logging.basicConfig(level=logging.INFO)



def filter_raw_data(sample_json: str) -> dict or None:
    """
        Filter a sample drawing based on the following rules:
        1. The image is recognized by the model
        2. The image is drawn in one stroke, totaling at least 10 pixel coordinates
        3. The image is drawn within the allowed time of 20s (20,000ms)
    """

    sample = json.loads(sample_json)

    if not sample.get("recognized", False):
        return None
    
    stroke_count = sample.get("drawing", [])
    if len(stroke_count) != 1:
        return None
    
    x_coord, y_coord, time = stroke_count[0]
    # Stroke length
    if len(x_coord) < 10:
        return None
    
    # Time length
    if time[-1] - time[0] >= 20000:
        return None

    return sample
    
    """
    [                               # Drawing
    [  // First stroke                # Stroke No.
        [x0, x1, x2, x3, ...],              # Stroke X
        [y0, y1, y2, y3, ...],              # Stroke Y
        [t0, t1, t2, t3, ...]               # Time t
    ],
    [  // Second stroke
        [x0, x1, x2, x3, ...],
        [y0, y1, y2, y3, ...],
        [t0, t1, t2, t3, ...]
    ],
    ... // Additional strokes
    ]

    stroke = sample.get("drawing", [])
        stroke = Drawing
        stroke[0] = Stroke No. 1
        stroke[0][0] = Stroke X
    """



def coords_to_matrix(
    stroke_coords: List[List[List[int]]],
    image_height: int = 28,
    image_width: int = 28,
) -> np.ndarray:
    """
        Extract the coordinates of each stroke of the image and translate them into 2D numpy array of grayscale pixels.
        By default, the image dimension is 28x28.
    """

    size = 256
    img = Image.new("L", (size, size), 0)
    sample_canvas = ImageDraw.Draw(img)

    for point in stroke_coords:
        x_coord, y_coord = point[0], point[1]
        point_pos = list(zip(x_coord, y_coord))

        if len(point_pos) > 1:
            sample_canvas.line(point_pos, fill=255, width=15)

    final_sample = img.resize((image_height, image_width), Image.Resampling.LANCZOS)
    return np.array(final_sample)
    


def load_and_split_raw_data(
    data_dir: str,
    class_list: List[str],
    raw_data_pattern: str = "full_raw_{}.ndjson",
    count_per_class: int = 25000,
    train_validate_test_ratio: Tuple[float] = (0.7, 0.1, 0.2)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
        Extract list of raw ndjson files from the data directory and split them into train and test sets based on the ratio.
        Class name should be a list of predefined classes involved in classification.
        The name of the raw data is expected as "full_raw_[Class name].ndjson
        By default, number of samples per class is 10000, and the split ratio for training set, validation set, and test set, is 70%, 10%, and 20% respectively.
        The ratio of validation set can be set to 0.0 to remove validation set.
    """

    if not class_list:
        raise ValueError("Function load_and_split_raw_data aborted: No class provided")

    train_ratio, val_ratio, test_ratio = train_validate_test_ratio
    if not np.allclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Function load_and_split_raw_data aborted: Ratio of each set does not sum to 100%")

    X_list, y_list = [], []
    label_map = {idx: name for idx, name in enumerate(class_list)}

    for idx, class_name in enumerate(class_list):
        file_name = raw_data_pattern.format(class_name)
        file_path = os.path.join(data_dir, file_name)
    
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing expected doodle database: {file_path}")

        category_images = []

        with open(file_path, "r") as f:
            for line in f:
                if len(category_images) >= count_per_class:
                    break
                    
                valid_sample = filter_raw_data(line)
                if valid_sample is None:
                    continue
                    
                strokes = valid_sample.get("drawing", [])
                matrix = coords_to_matrix(strokes)
                category_images.append(matrix)
            
        images_array = np.array(category_images).reshape(-1, 28, 28, 1).astype("float32") / 255.0
        labels_array = np.full((images_array.shape[0], 1), idx)

        logging.info(f"{len(images_array)} clean samples extracted for category: {class_name}")
        X_list.append(images_array)
        y_list.append(labels_array)

    X_total = np.concatenate(X_list, axis=0)
    y_total = np.concatenate(y_list, axis=0)

    remaining_ratio = train_ratio + val_ratio
    test_fraction = test_ratio / (remaining_ratio + test_ratio)
    
    X_remain, X_test, y_remain, y_test = train_test_split(
        X_total, y_total, test_size=test_fraction, random_state=42, stratify=y_total
    )

    if not val_ratio == 0.0:
        val_fraction = val_ratio / remaining_ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_remain, y_remain, test_size=val_fraction, random_state=42, stratify=y_remain
        )
        logging.info(f"Data Splitted with ratio: -> Train: {train_ratio}, Validation: {val_ratio}, Test: {test_ratio}")
    else:
        X_train, y_train = X_remain, y_remain
        X_val = np.empty((0, 28, 28, 1), dtype=np.float32)
        y_val = np.empty((0, 1), dtype=np.float32)
        logging.info(f"Data Splitted with ratio: Train: {train_ratio}, Test: {test_ratio}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_map