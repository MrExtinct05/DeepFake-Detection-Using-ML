from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import cv2

img_size = 224
batch_size = 64
max_seq_length = 20 
num_features = 2048

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Define helper function for video preprocessing
def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def preprocess_video(frames):
    frames = frames / 255.0
    frames = frames.reshape((1, frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
    return frames

def pretrain_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
    weights = "imagenet",
    include_top=False,
    pooling="avg",
    input_shape = (img_size,img_size,3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    
    inputs = keras.Input((img_size,img_size,3))
    preprocessed = preprocess_input(inputs)
    
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = pretrain_feature_extractor()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_seq_length,), dtype="bool")
    frame_features = np.zeros(shape=(1, max_seq_length, num_features), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq_length, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def predict_video(video_path):
    # Load the video frames
    frames = load_video(video_path)

    # Prepare the frames for the model
    frame_features, frame_mask = prepare_single_video(frames)

    # Make a prediction with the model
    predictions = model.predict([frame_features, frame_mask])

    # Return the prediction result
    return predictions[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Check if video file was uploaded
    if 'file' not in request.files:
        return "No video file found!"

    file = request.files['file']

    # Check if file is a video file
    if file.content_type not in ['video/mp4', 'video/avi']:
        return "File is not a video file!"

    # Save the uploaded video file to the server
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # # Load the video file
    # frames = load_video(filepath, max_frames=20, resize=(224, 224))
    # frames = preprocess_video(frames)

    # # Run the video frames through the model for classification
    # prediction = model.predict(frames)
    # prediction = prediction[0][0]
    
    prediction = predict_video(filepath)
    print("---------->>>>>>>>>>>",prediction)
    # Render the result page with the prediction
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
