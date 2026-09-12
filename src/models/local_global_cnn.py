import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_local_global_cnn(global_shape=(3197, 1), local_shape=(201, 1), learning_rate=1e-3):
    """
    Builds a dual-view (Global + Local) CNN architecture for exoplanet detection.
    Global Branch processes full light curve time series.
    Local Branch processes zoomed transit dip window.
    """
    # Global Branch
    input_global = layers.Input(shape=global_shape, name="input_global")
    g = layers.Conv1D(16, 5, activation='relu')(input_global)
    g = layers.MaxPooling1D(4)(g)
    g = layers.Conv1D(32, 5, activation='relu')(g)
    g = layers.MaxPooling1D(4)(g)
    g = layers.Conv1D(64, 5, activation='relu', name="global_last_conv")(g)
    g = layers.GlobalAveragePooling1D()(g)
    
    # Local Branch
    input_local = layers.Input(shape=local_shape, name="input_local")
    l = layers.Conv1D(16, 5, activation='relu')(input_local)
    l = layers.MaxPooling1D(2)(l)
    l = layers.Conv1D(32, 5, activation='relu', name="local_last_conv")(l)
    l = layers.GlobalAveragePooling1D()(l)
    
    # Merge Branches
    merged = layers.concatenate([g, l], name="concat_views")
    fc = layers.Dense(64, activation='relu')(merged)
    fc = layers.Dropout(0.3)(fc)
    output = layers.Dense(1, activation='sigmoid', name="output_layer")(fc)
    
    model = models.Model(inputs=[input_global, input_local], outputs=output, name="DualView_LocalGlobal_CNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )
    return model
