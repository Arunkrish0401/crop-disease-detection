import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model/crop_disease_model.keras"
DATASET_PATH = "data/PlantVillage"
IMG_SIZE = (128, 128)

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names (same order as training)
class_names = sorted(os.listdir(DATASET_PATH))

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    class_index = np.argmax(predictions)

    return class_names[class_index], round(confidence, 2)

# ==============================
# TEST PREDICTION (NO UI)
# ==============================
if __name__ == "__main__":
    test_image_path = "test_leaf.jpg"  # image in project root

    if not os.path.exists(test_image_path):
        print("❌ test_leaf.jpg not found in project root")
    else:
        disease, confidence = predict_disease(test_image_path)
        print("🦠 Predicted Disease:", disease)
        print("📊 Confidence:", confidence, "%")
