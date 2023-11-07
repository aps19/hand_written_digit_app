import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Title and description
st.title("MNIST Digit Classification")
st.write("This app allows you to train a Random Forest Classifier on the MNIST dataset.")

# Load the MNIST dataset
st.write("Loading the MNIST dataset...")
mnist = fetch_openml("mnist_784")
X, y = mnist.data, mnist.target
X = X / 255.0  # Normalize pixel values to [0, 1]

# Sidebar for hyperparameters
st.sidebar.header("Set Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators (Trees)", 10, 200, 100)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the Random Forest Classifier
st.write(f"Training the Random Forest Classifier with {n_estimators} trees...")
clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Display accuracy and confusion matrix
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Display classification report
st.subheader("Classification Report")
class_report = classification_report(y_test, y_pred)
st.text(class_report)

# Plot confusion matrix as a heatmap
st.subheader("Confusion Matrix Heatmap")
plt.figure(figsize=(8, 6))
st.write(plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest"))
st.pyplot()

# Display the model hyperparameters
st.subheader("Model Hyperparameters")
st.write(f"Number of Estimators (Trees): {n_estimators}")
st.write(f"Test Size: {test_size}")

