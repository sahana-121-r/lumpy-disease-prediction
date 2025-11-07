import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\Sahana R\\Cow-Lumpy-Prediction-CNN\\train.py\\cow_lumpy_notcow_model.h5")

# Image path
img_path = "C:\\Users\\Sahana R\\Cow-Lumpy-Prediction-CNN\\img\\sample.jpg8.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Define class labels (must match training order)
class_labels = ['healthy_cow', 'lumpy_cow', 'not_cow']

# Get predicted class and confidence
predicted_class = class_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Display output
if predicted_class == 'not_cow':
    print("⚠️ Invalid photo! Please upload a cow image.")
else:
    print(f"✅ Prediction: {predicted_class.replace('_', ' ').title()} ({confidence:.2f}% confidence)")







