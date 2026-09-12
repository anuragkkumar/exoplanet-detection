import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_1d_cnn(input_shape=(3197, 1), dropout_rate=0.3, learning_rate=1e-3):
    """
    Builds the tuned 3-layer 1D CNN for exoplanet light curve classification.
    
    Architecture:
        Conv1D(16, k=5, relu) -> MaxPool1D(4)
        Conv1D(32, k=5, relu) -> MaxPool1D(4)
        Conv1D(64, k=5, relu) -> MaxPool1D(4)
        Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
    """
    model = models.Sequential([
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape, name='conv1d_layer1'),
        layers.MaxPooling1D(pool_size=4, name='maxpool1d_1'),
        layers.Conv1D(32, kernel_size=5, activation='relu', name='conv1d_layer2'),
        layers.MaxPooling1D(pool_size=4, name='maxpool1d_2'),
        layers.Conv1D(64, kernel_size=5, activation='relu', name='conv1d_layer3'),
        layers.MaxPooling1D(pool_size=4, name='maxpool1d_3'),
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(dropout_rate, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ], name="1D_CNN_Exoplanet_Detector")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    return model
