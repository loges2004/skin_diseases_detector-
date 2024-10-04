from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from PIL import Image, ImageEnhance
import logging
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itsdangerous import URLSafeTimedSerializer
import joblib
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow_hub as hub
import os
import warnings
import numpy as np
from flask import session
import time
app = Flask(__name__)
app.secret_key = 'bb40cbdbba6658017c8b53e05580d865'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
mongo_uri = 'mongodb://localhost:27017/skin_disease_detector'
client = MongoClient(mongo_uri)
db_mongo = client.get_database('skin_disease_detector')
users_collection = db_mongo.users
history_collection = db_mongo.history

# Initialize URLSafeTimedSerializer
s = URLSafeTimedSerializer(app.secret_key)
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')



# Load the object detection model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load your pre-trained Keras model
keras_model_path =r'C:\Users\loges\java\.dist\skin_diseases\skin\skin_disease_detector\models\cnn_skin_disease_classifier.keras'
keras_model = tf.keras.models.load_model(keras_model_path)

# Create directories if they don't exist
os.makedirs('./static/uploads', exist_ok=True)

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        try:
            file = request.files['image']
            if file:
                # Save the uploaded image
                image_path = f"./static/uploads/{file.filename}"
                file.save(image_path)

                # Process the image
                processed_image_path, predicted_class = process_image(image_path)

                # Redirect to the result page to display the image and prediction
                return redirect(url_for('result', image_path=processed_image_path, predicted_class=predicted_class))
        except Exception as e:
            app.logger.error(f"Error in upload_image route: {str(e)}")
            return "Internal Server Error", 500
    return render_template('upload.html')



@app.route('/result')
def result():
    image_path = request.args.get('image_path')
    predicted_class = request.args.get('predicted_class')
    return render_template('result.html', image_path=image_path, predicted_class=predicted_class)



@app.route('/capture')
def capture():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Failed to capture image from webcam", 500

        image_path = "./static/uploads/webcam_capture.jpg"
        cv2.imwrite(image_path, frame)

        # Process the captured image
        processed_image_path, predicted_class = process_image(image_path)

        # Redirect to the result page to display the image and prediction
        return redirect(url_for('result', image_path=processed_image_path, predicted_class=predicted_class))
    except Exception as e:
        app.logger.error(f"Error in capture route: {str(e)}")
        return "Internal Server Error", 500

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import session, flash
from pymongo import MongoClient

# Load the model
keras_model_path = r'C:\Users\loges\java\.dist\skin_diseases\skin\skin_disease_detector\models\cnn_skin_disease_classifier.keras'
keras_model = tf.keras.models.load_model(keras_model_path)

# Define class names and unknown class
class_names = ['acne', 'healthy', 'psoriasis', 'vitiligo']
unknown_class = 'unknown'


# Function to resize images
def resize_image(image, size=(400, 400)):
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

# Preprocess the input image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize image to model's expected input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
    return img_array

# Predict the class of the input image
def predict_image_class(img_path, confidence_threshold=0.5):
    processed_image = preprocess_image(img_path)
    predictions = keras_model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest prediction
    predicted_label = class_names[predicted_class]  # Map index to class name
    confidence = predictions[0][predicted_class]  # Confidence of the prediction
    
    # If confidence is below the threshold, classify as unknown
    if confidence < confidence_threshold:
        return unknown_class, confidence
    
    return predicted_label, confidence

# Function to adjust image brightness
def adjust_brightness(image, alpha=1.5, beta=0):
    # Increase brightness by scaling and shifting pixel values
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function to check image quality
def check_image_quality(image, threshold=100.0):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Compute the Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    # Check if the variance is below the threshold
    return variance > threshold

# Function to process the image
def process_image(image_path):
    try:
        # Load the image
        image_rgb = cv2.imread(image_path)
        if image_rgb is None:
            flash("Image not found or unable to read image.", "error")
            return None, None
        
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Check image quality
        if not check_image_quality(image_rgb):
            flash("The uploaded image quality is too low. Please upload a clearer image.", "warning")
            return None, None
        
        # Resize image for the CNN model
        image_resized = resize_image(image_rgb, (400, 400))
    
        # Predict the class using the CNN model
        predicted_class, confidence = predict_image_class(image_path)
        
        # Adjust brightness
        image_resized_bright = adjust_brightness(image_resized, alpha=1.5, beta=30)
        
        # Convert image back to BGR for saving
        image_bgr_bright = cv2.cvtColor(image_resized_bright, cv2.COLOR_RGB2BGR)
        
        # Convert to uint8 before saving
        image_bgr_bright = np.clip(image_bgr_bright, 0, 255).astype('uint8')
        
        # Resize the image to save it
        processed_image_resized = resize_image(image_bgr_bright, (400, 400))
        processed_image_path = image_path.replace('uploads/', 'uploads/processed_')
        cv2.imwrite(processed_image_path, processed_image_resized)
        
        # Save prediction history if user is logged in
        if 'email' in session:
            email = session['email']
            history_collection.insert_one({
                'email': email,
                'image': processed_image_path,
                'prediction': predicted_class
            })
        
        return processed_image_path, predicted_class
    
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return None, None


@app.route('/')
def home():
    if 'email' in session:
        email = session['email']
        user = users_collection.find_one({'email': email})
        if user:
            username = user.get('username', 'Guest')  # Default to 'Guest' if username not found
            return render_template('index.html', username=username)
    return redirect(url_for('login'))



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check user in MongoDB
        user = users_collection.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['email'] = email
            session['username'] = user.get('username')  # Store username in session
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 400
    
    return render_template('login.html')
from pymongo.errors import DuplicateKeyError


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400

        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'Email already registered'}), 400

        hashed_password = generate_password_hash(password)

        try:
            users_collection.insert_one({'email': email, 'password': hashed_password, 'username': username})
        except DuplicateKeyError:
            return jsonify({'error': 'Username already taken'}), 400

        return jsonify({'message': 'Registration successful'}), 200

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))


@app.route('/history')
def history():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    email = session['email']
    history_entries = history_collection.find({'email': email})

    base_url = url_for('static', filename='uploads/')

    return render_template('history.html', history=history_entries, base_url=base_url)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    email = session['email']
    
    try:
        history_collection.delete_many({'email': email})
        flash('History cleared successfully!')
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        flash('An error occurred while clearing history.')
    
    return redirect(url_for('history'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']
    user = users_collection.find_one({'email': email})

    if request.method == 'POST':
        username = request.form['username']
        contact_number = request.form['contact_number']
        new_email = request.form['email']

        users_collection.update_one(
            {'email': email},
            {'$set': {'username': username, 'contact_number': contact_number, 'email': new_email}}
        )

        # Update session email if it has changed
        session['email'] = new_email

        flash('Profile updated successfully!')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

def send_email(subject, recipient, body):
    sender_email = 'your email'
    sender_password = 'yoyr password'  # Use a secure method to handle passwords

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Error sending email: {e}")


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = users_collection.find_one({'email': email})

        if not user:
            return jsonify({'status': 'error', 'message': 'Email not registered'}), 400

        token = s.dumps(email, salt='password-reset-salt')
        reset_link = url_for('reset_password', token=token, _external=True)

        subject = 'Password Reset Request'
        body = f'Please click the following link to reset your password: {reset_link}'
        send_email(subject, email, body)

        return jsonify({'status': 'success', 'message': 'Password reset link sent to your email'}), 200

    return render_template('forgot_password.html')


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        return jsonify({'status': 'error', 'message': 'The link is expired or invalid'}), 400

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return jsonify({'status': 'error', 'message': 'Passwords do not match'}), 400

        hashed_password = generate_password_hash(password)
        users_collection.update_one({'email': email}, {'$set': {'password': hashed_password}})
        return jsonify({'status': 'success', 'message': 'Password reset successful'}), 200

    return render_template('reset_password.html', token=token)


if __name__ == '__main__':
    app.run(debug=True)
