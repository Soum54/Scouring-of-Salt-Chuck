import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2  # OpenCV for image processing
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('plastic_model.h5')  # Update the path as needed

# Function to predict if the image contains plastic or not
def predict_image(img):
    # Preprocess the image
    img = img.resize((150, 150))  # Ensure it matches the model's input size
    img_array = np.array(img)
    
    # Convert to float32 for proper normalization
    img_array = img_array.astype('float32')  # Convert the image array to float32
    
    # Normalize the image
    img_array /= 255.0  # Scale pixel values between 0 and 1

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(img_array)

    # Output the result
    if prediction[0][0] > 0.5:  # Assuming sigmoid activation for binary classification
        result = "Plastic"
    else:
        result = "No Plastic"

    return result

# Function to detect edges in the image
def detect_edges(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(img_gray, 100, 200)  # Apply Canny edge detection

    # Save the edge-detected image
    edge_img_path = img_path.replace('.jpeg', '_edges.jpeg').replace('.jpg', '_edges.jpg').replace('.png', '_edges.png')  # Create a new filename
    cv2.imwrite(edge_img_path, edges)

    return edge_img_path

# Streamlit interface
st.title("Plastic Detection and Edge Detection")
st.write("Upload an image to check if it contains plastic and view the edge detection.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image with PIL
    img = Image.open(uploaded_file)

    # Predict plastic
    result = predict_image(img)
    
    # Display the uploaded image along with the prediction result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {result}")

    # If plastic is detected, perform edge detection
    if result == "Plastic":
        # Save the uploaded image temporarily for edge detection
        img_path = '/tmp/uploaded_image.jpg'
        img.save(img_path)

        # Perform edge detection
        edge_img_path = detect_edges(img_path)
        
        # Display edge-detected image
        edge_img = Image.open(edge_img_path)
        st.image(edge_img, caption="Edge Detection Result", use_column_width=True)

        # Clean up: delete the edge image after display (optional)
        os.remove(edge_img_path)
