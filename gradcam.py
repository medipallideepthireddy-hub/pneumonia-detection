"""
AI-Based Pneumonia Detection Using Attention-Enhanced CNN
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
"""

import os
import numpy as np
import cv2
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None


def get_target_layer_name(model):
    """
    Auto-detects the most appropriate convolutional layer for Grad-CAM.
    Prefers layers named 'target_conv_layer', 'attention_conv', or the last Conv2D.
    """
    conv_layer_names = []
    for layer in model.layers:
        if "target_conv" in layer.name:
            return layer.name
        if "conv" in layer.name.lower() and hasattr(layer, "output"):
            conv_layer_names.append(layer.name)

    if conv_layer_names:
        return conv_layer_names[-1]
    return None


def generate_gradcam_heatmap(model, img_array, target_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image array and target layer.

    Args:
        model: tf.keras.Model
        img_array: numpy array with shape (1, height, width, 3), normalized [0, 1]
        target_layer_name: str or None (auto-detects if None)
        pred_index: int or None (for binary classification, None defaults to single output)

    Returns:
        numpy 2D array (height, width) with values normalized to [0, 1], or None on error
    """
    if tf is None:
        return None

    if target_layer_name is None:
        target_layer_name = get_target_layer_name(model)

    if target_layer_name is None:
        print("[Grad-CAM Warning] No suitable convolutional layer found in model.")
        return None

    try:
        # Build gradient model connecting the input to the target conv layer & final output
        target_layer = model.get_layer(target_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            # Cast input to tensor
            img_tensor = tf.cast(img_array, tf.float32)
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)

            # In binary classification with 1 sigmoid output:
            if predictions.shape[-1] == 1:
                loss = predictions[0][0]
                # If predicting normal (loss < 0.5), we can invert to highlight normal features
                # or keep positive activation for pneumonia
                if pred_index == 0:
                    loss = 1.0 - loss
            else:
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]

        # Calculate gradients of the loss with respect to the convolutional feature map
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("[Grad-CAM Warning] Gradients evaluated to None.")
            return None

        # Compute guided average pooling across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature map by the computed gradients
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to keep only features that positively influence the target class
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize the heatmap between 0 and 1
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    except Exception as exc:
        print(f"[Grad-CAM Error] Heatmap generation failed: {exc}")
        return None


def overlay_heatmap_on_image(original_img_path, heatmap, output_path, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Superimposes a 2D Grad-CAM heatmap over the original radiograph image.

    Args:
        original_img_path: path to input image file
        heatmap: 2D numpy array [0, 1]
        output_path: path to save the overlaid image
        alpha: heatmap blending transparency factor (default: 0.45)
        colormap: OpenCV colormap constant (default: COLORMAP_JET)

    Returns:
        output_path if successful, or None
    """
    try:
        # Read the original image in BGR
        orig_img = cv2.imread(original_img_path)
        if orig_img is None:
            # Fallback using Pillow
            pil_img = Image.open(original_img_path).convert("RGB")
            orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = orig_img.shape[:2]

        # Resize heatmap to match original image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h))

        # Rescale heatmap to [0, 255] unsigned 8-bit integer
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Apply the colormap to produce colored heatmap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend the heatmap with the original radiograph
        superimposed_img = cv2.addWeighted(colored_heatmap, alpha, orig_img, 1.0 - alpha, 0)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to disk
        cv2.imwrite(output_path, superimposed_img)
        return output_path

    except Exception as exc:
        print(f"[Grad-CAM Error] Heatmap overlay saving failed: {exc}")
        return None
