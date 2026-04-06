import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_detector.keras")

model = load_model()

# ================= PREPROCESS =================
def preprocess(image):
    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)

    return image

# ================= GRAD-CAM =================
def generate_gradcam(model, image):

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(img, (224, 224))
    img_input = img_resized.astype(np.float32)
    img_input = (img_input / 127.5) - 1.0
    img_input = np.expand_dims(img_input, axis=0)

    # 🔥 BEST layer for your model
    last_conv_layer = model.get_layer("block_11_expand")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads

    weights = tf.reduce_mean(guided_grads, axis=(1, 2))

    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)
    cam = cam[0].numpy()

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # 🔥 Smooth
    cam = cv2.GaussianBlur(cam, (15, 15), 0)

    # 🔥 Strong threshold
    threshold = 0.6
    cam[cam < threshold] = 0

    # Normalize again
    cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.85, heatmap, 0.25, 0)

    return overlay

# ================= UI =================
st.title("🔥 Grad-CAM Visualization")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original MRI", use_container_width=True)

    with col2:
        gradcam = generate_gradcam(model, image)
        st.image(gradcam, caption="Grad-CAM Output", use_container_width=True)