import pytest
import tensorflow as tf
from src.model import build_model



def test_build_model_valid():
    """Test if the build_model run successfully under the correct configurations"""
    batch_size = 16
    class_count = 4
    h, w, c = 28, 28, 1

    model = build_model(
        class_count = class_count,
        input_height = h,
        input_width = w,
        input_channel = c,
        use_augmentation = True,
        conv_set = True,
        conv_filter = 4,
        conv_kernel = 3,
        use_batch_normalization = True,  
        use_dropout = True,
        dropout_rate = 0.2,
        use_max_pooling_2d = True,
        final_dense_neuron = 4
    )

    mock_input = tf.random.normal((batch_size, h, w, c))
    prediction = model(mock_input)

    assert prediction.shape == (batch_size, class_count)



def test_build_model_invalid_class():
    """Test if the build_model aborts when class_count receives an invalid value"""
    batch_size = 16
    class_count = 1
    h, w, c = 28, 28, 1

    with pytest.raises(ValueError, match = "Provided class count is not applicable for classification task."):
        build_model(
            class_count = class_count,
            input_height = h,
            input_width = w,
            input_channel = c,
            use_augmentation = True,
            conv_set = True,
            conv_filter = 4,
            conv_kernel = 3,
            use_batch_normalization = True,  
            use_dropout = True,
            dropout_rate = 0.2,
            use_max_pooling_2d = True,
            final_dense_neuron = 4
        )

def test_build_model_invalid_kernel():
    """Test if the build_model aborts when class_count receives an invalid value"""
    batch_size = 16
    class_count = 4
    h, w, c = 28, 28, 1

    with pytest.raises(ValueError, match = "Kernel sizes should be positive odd integers."):
        build_model(
            class_count = class_count,
            input_height = h,
            input_width = w,
            input_channel = c,
            use_augmentation = True,
            conv_set = True,
            conv_filter = 4,
            conv_kernel = 4,
            use_batch_normalization = True,  
            use_dropout = True,
            dropout_rate = 0.2,
            use_max_pooling_2d = True,
            final_dense_neuron = 4
        )