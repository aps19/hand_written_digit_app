import streamlit as st
from skorch import NeuralNetClassifier
import random
import torch
import os
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skorch.callbacks import EpochScoring
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
import pickle
from torchvision import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Define model as a global variable to make it accessible for all pages
model = None


def create_cnn_model(learning_rate, dropout, device, hidden_units, optimizer,epochs):
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

def load_mnist_data(batch_size, test_split):
    st.subheader("Loading Dataset")
    st.text("1. Downloading MNIST Dataset")
    
    data_dir = os.path.expanduser("~/data")

    # Load the MNIST dataset using fetch_openml
    mnist = fetch_openml('mnist_784', data_home=data_dir, as_frame=False, cache=True, parser='auto')
    
    st.text("2. Dataset Downloaded")

    # Extract the data and labels
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255.0  # Normalize the pixel values to [0, 1]
    X = X.reshape(-1, 1, 28, 28)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    st.text("3. Randomly split training and test datasets.")

    # Display a 3x3 matrix of example images
    st.subheader("Example Images")
    example_indices = np.random.choice(len(X_train), 9, replace=False)
    example_images = X_train[example_indices]
    example_labels = y_train[example_indices]

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            ax = axes[i, j]
            ax.imshow(example_images[index][0], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Label: {example_labels[index]}", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    return X_train, X_test, y_train, y_test


def training_and_evaluation_app():
    st.title("Training and Evaluation Page")
    st.write("On this page, you can set hyperparameters, train the model, and evaluate its performance.")

    # Schema of the Training Process
    st.subheader("Training Process")
    st.write("1. **Loading the MNIST Dataset**: The MNIST dataset is loaded and split into training and testing sets.")
    st.write("2. **Creating the CNN Model**: A Convolutional Neural Network (CNN) is created with user-defined hyperparameters.")
    st.write("3. **Training the Model**: The model is trained with the specified hyperparameters, such as learning rate, dropout rate, and hidden units.")
    
    # Hyperparameters input
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
    dropout = st.sidebar.number_input("Dropout Rate", 0.0, 0.9, 0.5, 0.1)
    hidden_units = st.sidebar.number_input("Hidden Units", 10, 500, 100, 10)
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD"])
    epochs = st.sidebar.number_input("Number of Epochs",1,100,10,1)

    # Allow the user to input the batch size
    batch_size = st.sidebar.number_input("Batch Size", 1, 512, 64, 1)
    test_split = st.sidebar.number_input("Test data size",0.01,0.5,0.2,0.01)

    X_train, X_test, y_train, y_test  = load_mnist_data(batch_size, test_split)  # Pass the batch size to the data loader
    
    optimizer = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD

    # Train the model
    if st.button("Train Model"):
        st.write(f"Training the model with learning rate: {learning_rate}, dropout: {dropout}, hidden units: {hidden_units}, batch size: {batch_size}...")

        global model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_cnn_model(learning_rate, dropout, device, hidden_units, optimizer,epochs)
        
        # Training results
        st.subheader("Training Results")
        
        st.write(model.fit(X_train, y_train))  # Train for one epoch

        # Evaluate the model on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write("Test Accuracy:", accuracy)
        
        st.subheader("Model Evaluation")
        st.write("Evaluating the model...")
        
        # Evaluate the model on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write("Test Accuracy:", accuracy)

        conf_matrix = confusion_matrix(y_true, y_pred)
        st.subheader("Confusion Matrix")
        st.write(conf_matrix)

        class_report = classification_report(y_true, y_pred)
        st.subheader("Classification Report")
        st.text(class_report)

        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
        st.subheader("Confusion Matrix Heatmap")
        st.pyplot(plt)
            
        if st.button("Save Model"):
            # Generate a timestamp
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Define the filename with the timestamp
            model_save_path = f"complete_model_{current_time}.pkl"
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)