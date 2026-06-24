import os
import json
from typing import List, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
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