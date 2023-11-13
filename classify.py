# Import necessary libraries
import streamlit as st
import torch
from PIL import Image
import numpy as np

from utility import create_cnn_model


# Function to load the trained model
def load_trained_model(model_path):
    # Load the model from the specified path
    model = torch.load(model_path)
    # Set the model to evaluation mode
    model.eval()
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Normalize pixel values to [0, 1]
    image = np.array(image) / 255.0
    # Add batch dimension and channel dimension
    image = image[np.newaxis, np.newaxis, :, :]
    # Convert to a PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32)
    return image


# Function to perform image classification
def classify_image(model, image):
    with torch.no_grad():
        # You should use the 'predict' method to make predictions
        predicted_classes = model.predict(image)
        # Get the predicted class (digit)
        predicted_class = predicted_classes[0]  # Assuming there's a single prediction
    return predicted_class


# Create a Streamlit app page for image classification
def image_classification_app():
    st.title("Image Classification Page")
    st.write("Upload an image for digit classification.")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        image = preprocess_image(image)

        # Define a new model instance with the same architecture
        model = create_cnn_model()
        model.initialize()  # Important: Initialize the model

        # Load the saved parameters, optimizer state, and history
        model.load_params(
            f_params='digit_model_params.pkl',
            f_optimizer='digit_optimizer_state.pkl',
            f_history='digit_training_history.json'
        )

        st.write("Model loaded successfully!")
        # Classify the uploaded image
        predicted_class = classify_image(model, image)

        st.subheader("Image Classification Result")
        st.write(f"Predicted Digit: {predicted_class}")
