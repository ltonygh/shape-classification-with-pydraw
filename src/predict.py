import os
import logging
import json
import numpy as np
import tensorflow as tf
from src.data_loader import coords_to_matrix

logging.basicConfig(level=logging.INFO)



labels = {
    0: "circle",
    1: "square",
    2: "star",
    3: "triangle"
}
model_path = "models/best_shape_model.keras"
selected_model = None

def get_model():
    """
        Lazy loads the model from disk and standby in cache.
    """

    global selected_model
    if selected_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Function get_model aborted: Model not found at path {model_path}."
                f"To create and train a model, run model_tune.py with 'python3 src/model_tune.py'."    
            )
        selected_model = tf.keras.models.load_model(model_path)
        logging.info(f"Model weights loaded from {model_path}")
    return selected_model



def predict_drawing(strokes: list) -> tuple:
    """
        Predicts the drawn doodle when the user draws on a 28x28 canvas to one of four labels.
        For more accurate predictions, an identifiable class of drawing should reach at least 95% confidence. Otherwise, classify as "Unknown"
    """

    if not strokes or len(strokes) == 0:
        return "Unknown", 0.0
    
    model = get_model()

    pixel_matrix = coords_to_matrix(strokes, image_height=28, image_width=28)

    processed_input = pixel_matrix.astype("float32") / 255.0
    processed_input = np.expand_dims(processed_input, axis=(0, -1))

    predict = model.predict(processed_input, verbose=0)[0]

    best_match_idx = int(np.argmax(predict))
    confidence_score = float(predict[best_match_idx]) * 100.0

    if confidence_score < 95.0:
        return "Unknown", confidence_score
        
    predicted_shape = labels.get(best_match_idx, "Unknown")
    return predicted_shape, confidence_score