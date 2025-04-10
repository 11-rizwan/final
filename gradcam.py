import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
import os

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, output_path="static/gradcam.jpg", alpha=0.4):
    original_img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    output_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)
    return output_path
