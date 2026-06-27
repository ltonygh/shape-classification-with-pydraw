import pytest
import numpy as np
from src.data_loader import load_and_split_raw_data



def test_load_and_split_raw_data():
    class_list = ["circle", "square", "star", "triangle"]
    count_per_class = 100

    X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_and_split_raw_data(
        data_dir = "data/raw_data",
        class_list = class_list,
        count_per_class = count_per_class,
        train_validate_test_ratio=(0.7, 0.1, 0.2)
    )

    total_samples = len(X_train) + len(X_val) + len(X_test)

    assert total_samples == len(class_list) * count_per_class

    assert np.isclose(len(X_train), total_samples * 0.7, atol=2)
    assert np.isclose(len(X_val), total_samples * 0.1, atol=2)
    assert np.isclose(len(X_test), total_samples * 0.2, atol=2)

    assert X_train.shape[1:] == (28, 28, 1)
    assert X_val.shape[1:] == (28, 28, 1)
    assert X_test.shape[1:] == (28, 28, 1)

    assert y_train.shape == (X_train.shape[0], 1)
    assert y_val.shape == (X_val.shape[0], 1)
    assert y_test.shape == (X_test.shape[0], 1)
    
    assert np.min(X_train) >= 0.0
    assert np.max(X_train) <= 1.0

    assert X_train.dtype == np.float32
    assert y_train.dtype in [np.int32, np.int64, np.float32]

    assert len(label_map) == 4
    assert len(np.unique(y_train)) == 4