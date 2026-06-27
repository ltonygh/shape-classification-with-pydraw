import os
import logging
import keras
import keras_tuner as kt
from keras_tuner import BayesianOptimization
from src.data_loader import load_and_split_raw_data 
from src.model import build_model

logging.basicConfig(level=logging.INFO)



class TuneHyperModel(kt.HyperModel):
    def __init__(self, class_count: int):
        super().__init__()
        self.class_count = class_count

    def build(self, hp) -> keras.Sequential:
        hp_conv_set = hp.Int("Number of Conv2D set", min_value = 1, max_value = 6, step = 1)
        hp_conv_filter = hp.Int("Number of filters in Conv2D layer (2^n)", min_value = 1, max_value = 4, step = 1)
        hp_conv_kernel = hp.Int("Kernel size in Conv2D layer", min_value = 3, max_value = 7, step = 2)

        hp_dropout = hp.Boolean("Use Dropout layer")
        hp_dropout_rate = hp.Float('Dropout rate', min_value = 0.1, max_value = 0.5, step = 0.05)

        hp_final_dense_neuron = hp.Int("Number of neurons in the final dense layer (2^n)", min_value = 3, max_value = 7, step = 1)
    
        model = build_model(
            class_count = self.num_classes,
            input_height = 28,
            input_width = 28,
            input_channel = 1,
            use_augmentation = hp.Boolean('Use augmentation layer set'),
            conv_set = hp_conv_set,
            conv_filter = hp_conv_filter,
            conv_kernel = hp_conv_kernel,
            use_batch_normalization = hp.Boolean("Use BatchNormalization layer"),
            use_dropout = hp_dropout,
            dropout_rate = hp_dropout_rate,
            use_max_pooling_2d = hp.Boolean("Use MaxPooling2D layer"),
            final_dense_neuron = hp_final_dense_neuron
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
def run_tuning_pipeline():
    """Main execution block to fetch data datasets and launch the tuning cluster."""
    dir = "data/raw_data"
    class_type = ["circle", "square", "triangle", "star"]
    class_count = len(class_type)

    X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_and_split_raw_data(
        data_dir = dir,
        class_name = class_type,
        count_per_class=10000,
        train_validate_test_ratio=(0.7, 0.1, 0.2)
    )

    hypermodel = TuneHyperModel(class_count = class_count)

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective='val_loss',
        max_trials=10,
        directory='models/tuning_logs',
        project_name='shape_doodle_tuning',
        overwrite=True
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )

    tuner.search(
        X_train, y_train, 
        epochs=30, 
        validation_data=(X_val, y_val), 
        callbacks=[early_stopping]
    )

    best_models = tuner.get_best_models(num_models=1)
    if best_models:
        best_model = best_models[0]
        # Ensure deployment weights folder exists and save model
        os.makedirs("models", exist_ok=True)
        best_model.save("models/best_shape_model.keras")
        logging.info("Tuning complete. Absolute best production model exported to models/best_shape_model.keras")
    else:
        logging.error("Tuning failed to recover a valid optimal candidate model.")
    
if __name__ == "__main__":
    run_tuning_pipeline()