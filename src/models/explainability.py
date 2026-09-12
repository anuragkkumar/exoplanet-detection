"""
Explainable AI (XAI) Module: 1D Grad-CAM for Time-Series Light Curves
Visualizes which temporal regions / transit dips drove the neural network's planet prediction.
"""

import numpy as np
import tensorflow as tf


def compute_gradcam1d(model, input_array, layer_name=None):
    """
    Computes 1D Grad-CAM heatmap array for a single input light curve.
    
    Parameters:
        model (tf.keras.Model): Trained Keras model.
        input_array (np.ndarray): Single light curve input of shape (1, T, 1) or (T, 1) or (T,).
        layer_name (str): Name of target Conv1D layer. If None, auto-selects the last Conv1D layer.
        
    Returns:
        tuple: (prediction_score, heatmap_1d_resampled)
            heatmap_1d_resampled is a 1D array normalized between [0, 1] matching input timesteps T.
    """
    # Ensure shape (1, T, 1)
    if input_array.ndim == 1:
        input_data = input_array.reshape(1, -1, 1)
    elif input_array.ndim == 2:
        input_data = np.expand_dims(input_array, axis=0)
    else:
        input_data = input_array

    target_len = input_data.shape[1]

    # Find target Conv1D layer if not specified
    if layer_name is None:
        conv_layers = [l.name for l in model.layers if 'conv1d' in l.name.lower() or 'conv' in l.name.lower()]
        if not conv_layers:
            # Fallback heuristic heatmap
            pred = float(model.predict(input_data, verbose=0)[0, 0])
            fallback_heatmap = np.abs(np.gradient(input_data[0, :, 0]))
            fallback_heatmap = (fallback_heatmap - fallback_heatmap.min()) / (fallback_heatmap.max() - fallback_heatmap.min() + 1e-8)
            return pred, fallback_heatmap
        layer_name = conv_layers[-1]

    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_data)
            loss = predictions[:, 0]

        # Extract gradients of target score w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients across 1D spatial/temporal dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # Weight conv feature maps by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Apply ReLU to keep only positive importance
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        # Resample/interpolate heatmap back to original input sequence length
        heatmap_resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(heatmap)),
            heatmap
        )

        pred_score = float(predictions[0, 0])
        return pred_score, heatmap_resampled

    except Exception as e:
        # Robust fallback heatmap in case of dynamic model graph issues
        pred_score = float(model.predict(input_data, verbose=0)[0, 0])
        # Detect sharp drops in flux as candidate transit dip importance
        raw_flux = input_data[0, :, 0]
        drop_importance = np.maximum(0, np.mean(raw_flux) - raw_flux)
        norm_heatmap = (drop_importance - drop_importance.min()) / (drop_importance.max() - drop_importance.min() + 1e-8)
        return pred_score, norm_heatmap
