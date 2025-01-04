from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
MODEL_PATH = 'age_gender_model.keras'
model = load_model(MODEL_PATH)

# Define gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# Function to preprocess the image for prediction
def preprocess_image(image):
    """
    Preprocess the uploaded image by converting to grayscale, resizing, and normalizing.
    Ensures that the image shape is consistent for prediction.
    """
    # Convert the image to grayscale and resize to the input shape (128, 128)
    image = image.convert("L").resize((128, 128))
    # Convert the image to a NumPy array and normalize
    image = img_to_array(image) / 255.0
    # Reshape to match the input shape for the model (1, 128, 128, 1)
    return image.reshape(1, 128, 128, 1)

# Helper function to delete old uploaded files
def delete_previous_uploads(upload_folder):
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Delete previously uploaded photos
        delete_previous_uploads(app.config['UPLOAD_FOLDER'])

        # Check if an image file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file to a folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Open and preprocess the image
            img = Image.open(filepath)
            processed_img = preprocess_image(img)

            # Make predictions using the pre-trained model
            predictions = model.predict(processed_img)

            # Extract predicted gender and age
            predicted_gender = gender_dict[round(predictions[0][0][0])]
            predicted_age = round(predictions[1][0][0])

            # Pass results to the template
            return render_template('index.html', 
                                   uploaded_image=filepath, 
                                   predicted_gender=predicted_gender, 
                                   predicted_age=predicted_age)
    
    return render_template('index.html')

def get_public_ip():
    try:
        # Run the curl command to get public IP
        result = subprocess.check_output(["curl", "-s", "ifconfig.me"], text=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

if __name__ == '__main__':
    #public_ip = get_public_ip()
    #print(f"External access: http://{public_ip}:80")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=80, debug=True)
    #app.run(debug=True)


