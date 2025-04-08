
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model('model/pneumonia_model.h5')
last_conv_layer_name = 'top_conv'  # EfficientNetB5 last conv layer

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def calculate_infection_percentage(heatmap):
    threshold = 0.6
    infected_area = np.sum(heatmap > threshold)
    total_area = heatmap.shape[0] * heatmap.shape[1]
    return round((infected_area / total_area) * 100, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    name = request.form.get('name')
    age = request.form.get('age')

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        result = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        percentage = 0
        if result != "Normal":
            img_for_gradcam = get_img_array(file_path, size=(224, 224))
            heatmap = make_gradcam_heatmap(img_for_gradcam, model, last_conv_layer_name)
            save_and_display_gradcam(file_path, heatmap)
            percentage = calculate_infection_percentage(heatmap)

        return render_template("result.html", name=name, age=age, result=result,
                               confidence=confidence, percentage=percentage)

if __name__ == '__main__':
    app.run(debug=True)
