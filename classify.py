# Import necessary libraries
import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle

def create_cnn_model(learning_rate=0.01, dropout=0.5, device='cpu', hidden_units=100, optimizer='Adam',epochs=10):
    class Cnn(nn.Module):
        def __init__(self, dropout=0.5):
            super(Cnn, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
            self.fc2 = nn.Linear(100, 10)
            self.fc1_drop = nn.Dropout(p=dropout)

        def forward(self, x):
            x = torch.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

            # flatten over channel, height and width = 1600
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

            x = torch.relu(self.fc1_drop(self.fc1(x)))
            x = torch.softmax(self.fc2(x), dim=-1)
            return x

    model = NeuralNetClassifier(
        Cnn,
        max_epochs=epochs,
        lr=learning_rate,
        optimizer=optimizer,
        device=device,
    )

    
    return model


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
    # Convert to a PyTorch tensor and add batch dimension
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

# Function to perform image classification
def classify_image(model, image):
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image)
        # Get the predicted class (digit)
        predicted_class = outputs.argmax().item()
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
            f_params='model_params.pkl',
            f_optimizer='optimizer_state.pkl',
            f_history='training_history.json'
        )

        st.write("Model loaded successfully!")
        # Classify the uploaded image
        predicted_class = classify_image(model, image)

        st.subheader("Image Classification Result")
        st.write(f"Predicted Digit: {predicted_class}")
