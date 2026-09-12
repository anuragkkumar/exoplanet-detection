import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def _residual_block_1d(x, filters, kernel_size=5, stride=1, name_prefix="resblock"):
    """1D Residual Block with skip connection."""
    shortcut = x
    
    # First Conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)
    
    # Second Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # Match shortcut dimensions if necessary
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', name=f"{name_prefix}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)
        
    x = layers.add([x, shortcut], name=f"{name_prefix}_add")
    x = layers.Activation('relu', name=f"{name_prefix}_out_relu")(x)
    return x


def build_1d_resnet(input_shape=(3197, 1), learning_rate=1e-3):
    """
    Builds a 1D Residual Network (ResNet1D) for time-series classification.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Initial Convolution
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', name="init_conv")(inputs)
    x = layers.BatchNormalization(name="init_bn")(x)
    x = layers.Activation('relu', name="init_relu")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="init_pool")(x)
    
    # Residual Blocks
    x = _residual_block_1d(x, 32, name_prefix="block1")
    x = _residual_block_1d(x, 64, stride=2, name_prefix="block2")
    x = _residual_block_1d(x, 128, stride=2, name_prefix="block3_last_conv")
    
    # Global Pooling & Head
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation='relu', name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="1D_ResNet_Exoplanet_Detector")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    return model
