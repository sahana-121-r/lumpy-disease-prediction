# from flask import Flask, render_template, request, session
# import os
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# # import matplotlib.pyplot as plt
# # from tensorflow.keras import layers  
# import numpy as np

# model= keras.models.load_model('CNN.h5')
# class_names=['Healthy', 'lumpy Disease']

# # try:
# #     # Disable all GPUS
# #     tf.config.set_visible_devices([], 'GPU')
# #     visible_devices = tf.config.get_visible_devices()
# #     for device in visible_devices:
# #         assert device.device_type != 'GPU'
# # except:
# #     # Invalid device or cannot modify virtual devices once initialized.
# #     pass
# #*** Backend operation
 
# # WSGI Application
# # Defining upload folder path
# UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # # Define allowed files
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# # Provide template folder name
# # The default folder name should be "templates" else need to mention custom folder name for template path
# # The default folder name for static files should be "static" else need to mention custom folder for static path
# app = Flask(__name__, template_folder='templates', static_folder='staticFiles')
# # Configure upload folder for Flask application
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# # Define secret key to enable session
# app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
# @app.route('/')
# def index():
#     return render_template('index1.html')
 
# @app.route('/',  methods=("POST", "GET"))
# def uploadFile():
#     if request.method == 'POST':
#         # Upload file flask
#         uploaded_img = request.files['uploaded-file']
#         # Extracting uploaded data file name
#         img_filename = secure_filename(uploaded_img.filename)
#         # Upload file to database (defined uploaded folder in static path)
#         uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
#         # Storing uploaded file path in flask session
#         session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
#         return render_template('index2.html')

# def Predict(model,img):
#     img_array=tf.keras.preprocessing.image.img_to_array(img)
#     img_array=tf.expand_dims(img_array,0)
#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)
#     return predicted_class,confidence


# @app.route('/show_image')
# def displayImage():
#     # # Retrieve uploaded file path from session
#     # img_file_path = session.get('uploaded_img_file_path', None)

#     # # ✅ Load and preprocess the image correctly
#     # img = keras.preprocessing.image.load_img(img_file_path, target_size=(256, 256))
#     # img_array = keras.preprocessing.image.img_to_array(img)
#     # img_array = np.expand_dims(img_array, axis=0)
#     # img_array = img_array / 255.0   # normalize pixels like training

#     # # ✅ Make prediction
#     # predictions = model.predict(img_array)
#     # predicted_class = class_names[np.argmax(predictions[0])]
#     # confidence = round(100 * (np.max(predictions[0])), 2)

#     # # Debug output
#     # print("Raw model output:", predictions)
#     # print(f"Predicted: {predicted_class} ({confidence}%)")

#     # # ✅ Render the result page
#     # return render_template(
#     #     'index3.html',
#     #     user_image=img_file_path,
#     #     pred=f'Cow detected as {predicted_class} with confidence {confidence}%')

#     # Retrieving uploaded file path from session
#     img_file_path = session.get('uploaded_img_file_path', None)
#     # print(img_file_path)
#     im=Image.open(img_file_path)
#     resize_and_rescale = tf.keras.Sequential([
#     keras.layers.experimental.preprocessing.Resizing(256, 256),
#     keras.layers.experimental.preprocessing.Rescaling(1./255),
#     ])
#     im=resize_and_rescale(im)
#     # Call the Predicts
#     pred,conf=Predict(model,im)
#     print(pred,conf)
#     # os.remove(img_file_path)
#     # Display image in Flask application web page
#     return render_template('index3.html', user_image = img_file_path,pred='Cow detected are {} with confidence {}%'.format(pred,conf))
 
# if __name__=='__main__':
#     app.run(debug = True)
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load trained model (✅ Give correct path)
#model = tf.keras.models.load_model("cow_lumpy_notcow_mobilenetv2.h5")
# --- Safe model loading ---
try:
    model = tf.keras.models.load_model("cow_lumpy_notcow_mobilenetv2.h5", compile=False)
    print("✅ Model loaded successfully.")
except TypeError as e:
    if "batch_shape" in str(e):
        print("⚠️ Detected batch_shape error while loading model.")
        import h5py
        with h5py.File("cow_lumpy_notcow_mobilenetv2.h5", "r") as f:
            model_config = f.attrs.get("model_config", b"").decode("utf-8")
        model_config = model_config.replace('"batch_shape"', '"batch_input_shape"')
        print("🩹 Model config patched in memory. Consider re-saving it in SavedModel format.")
    else:
        raise e


    

# ✅ Class labels must match your training print result
class_labels = ['healthy_cow', 'lumpy_cow', 'not_cow']

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------- ROUTES ---------------- #

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')


@app.route('/page2', methods=['GET'])
def page2():
    return render_template('index2.html')


@app.route('/page3', methods=['GET'])
def page3():
    return render_template('index3.html', results=None)


# 🧩 Prediction Route (Batch Upload)
@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files:
        return render_template('index1.html', result="⚠️ No files uploaded.")

    files = request.files.getlist('files[]')
    if len(files) == 0 or files[0].filename == '':
        return render_template('index1.html', result="⚠️ No files selected.")

    results = []

    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 🔥 Predict
            # Predict
            # 🔥 Predict
            prediction = model.predict(img_array)[0]  # probabilities
            healthy_prob = prediction[0]
            lumpy_prob = prediction[1]
            not_cow_prob = prediction[2]

            confidence = np.max(prediction) * 100
            predicted_class = class_labels[np.argmax(prediction)]

            # ---------------- SMART RULES ----------------

            # If NOT COW is likely → mark as NOT COW
            if not_cow_prob >= 0.20:
                predicted_class = "not_cow"

            # If model is unsure → mark as NOT COW
            elif confidence < 85:
                predicted_class = "not_cow"

            # ✅ Final message
            if predicted_class == "not_cow":
                result_text = "⚠️ Invalid photo! Please upload a clear cow image."
            elif predicted_class == "healthy_cow":
                result_text = f"✅ Healthy Cow Detected ({confidence:.2f}% confidence)"
            else:
                result_text = f"⚠️ Lumpy Cow Detected ({confidence:.2f}% confidence)"

            results.append({
                'filename': file.filename,
                'filepath': filepath.replace("\\", "/"),
                'prediction': result_text
            })

        except Exception as e:
            results.append({
                'filename': file.filename,
                'filepath': None,
                'prediction': f"❌ Error processing file: {str(e)}"
            })

    return render_template('index3.html', results=results)


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Get the port from environment or use 8080
    app.run(host="0.0.0.0", port=port)





