from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
import cv2
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['GRADCAM_FOLDER'] = 'static/gradcam/'
app.config['PDF_FOLDER'] = 'static/reports/'

model = load_model('pneumonia_efficientnetb5_model.h5')
last_conv_layer_name = 'top_conv'

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

def generate_pdf(name, age, diagnosis, confidence, percentage, image_path):
    os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)
    pdf_path = os.path.join(app.config['PDF_FOLDER'], f"{name}_report.pdf")
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, f"Pneumonia Detection Report")
    c.drawString(100, 780, f"Name: {name}")
    c.drawString(100, 760, f"Age: {age}")
    c.drawString(100, 740, f"Diagnosis: {diagnosis}")
    c.drawString(100, 720, f"Confidence: {confidence}%")
    if diagnosis != 'Normal':
        c.drawString(100, 700, f"Infection Percentage: {percentage}%")
    if os.path.exists(image_path):
        c.drawImage(image_path, 100, 400, width=300, height=300)
    c.save()
    return pdf_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    name = request.form.get('name')
    age = request.form.get('age')

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        result = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        percentage = 0
        gradcam_path = None

        if result != "Normal":
            img_for_gradcam = get_img_array(file_path, size=(224, 224))
            heatmap = make_gradcam_heatmap(img_for_gradcam, model, last_conv_layer_name)
            gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], f"gradcam_{filename}")
            os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)
            save_and_display_gradcam(file_path, heatmap, output_path=gradcam_path)
            percentage = calculate_infection_percentage(heatmap)

        pdf_path = generate_pdf(name, age, result, confidence, percentage, gradcam_path if gradcam_path else file_path)

        return render_template("result.html", name=name, age=age, result=result,
                               confidence=confidence, percentage=percentage, 
                               image_path=gradcam_path if gradcam_path else None,
                               pdf_path=pdf_path)

@app.route('/download/<filename>')
def download_report(filename):
    return send_file(os.path.join(app.config['PDF_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
