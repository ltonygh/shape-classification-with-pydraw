import pytest
import tensorflow as tf
from src import predict
from src.predict import get_model, predict_drawing



@pytest.fixture(autouse=True)
def reset_global_model():
    """
        Reset the global cache before running tests
    """
    predict.selected_model = None
    yield



@pytest.fixture
def mock_trained_model(tmp_path, monkeypatch):
    """
        Generates a simple model file as the temporary model used in model_path.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "test_model.keras"
    
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    model.save(str(model_file))
    
    monkeypatch.setattr(predict, "model_path", str(model_file))
    return str(model_file)



def test_predict_drawing(mock_trained_model):
    """
        Test if the prediction result matches one of four classes or as unknown, and return a confidence value.
    """

    dummy_drawing = [
        [
            [3, 4, 5, 6, 7, 6, 6, 5, 5, 5, 4, 4, 3], # X coords
            [3, 3, 3, 3, 3, 4, 5, 6, 7, 6, 5, 4, 3], # Y coords
            [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13], # Timestamps
        ]
    ]
    
    shape, confidence = predict.predict_drawing(dummy_drawing)
    
    assert isinstance(shape, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 100.0
    assert shape in ["circle", "square", "star", "triangle", "Unknown"]



def test_predict_drawing_empty_canvas():
    """
        Test if the model can handle empty canvas.
    """
    
    shape, confidence = predict.predict_drawing([])
    assert shape == "Unknown"
    assert confidence == 0.0