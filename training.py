import streamlit as st
from skorch import NeuralNetClassifier
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
# Define model as a global variable to make it accessible for all pages
model = None

st.set_page_config(layout="wide")
def create_cnn_model(learning_rate, dropout, device, hidden_units, optimizer, epochs):
    class Cnn(nn.Module):
        def __init__(self, dropout=0.5):
            super(Cnn, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.fc1 = nn.Linear(1600, 100)  # 1600 = number channels * width * height
            self.fc2 = nn.Linear(100, 10)
            self.fc1_drop = nn.Dropout(p=dropout)

        def forward(self, x):
            x = torch.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

            # flatten over channel, height, and width = 1600
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

            x = torch.relu(self.fc1_drop(self.fc1(x)))
            x = torch.softmax(self.fc2(x), dim=-1)
            return x

    # Convert the model parameters to the same data type as the input data
    model = NeuralNetClassifier(
        Cnn,
        max_epochs=epochs,
        lr=learning_rate,
        optimizer=optimizer,
        device=device,
        criterion=torch.nn.CrossEntropyLoss,  # Specify the criterion explicitly
    )

    return model


def load_mnist_data(batch_size, test_split):
    st.subheader("Loading Dataset")
    # Load MNIST dataset
    data = np.load('mnist_dataset.npz', allow_pickle=True)
    X, y = data['data'], data['target']

    # Preprocess the data
    X /= 255.0
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
    st.write("On this page, you can experiment with different hyperparameters, train the model, and evaluate its performance.")

    # Schema of the Training Process
    st.subheader("Training process in brief:")
    st.write("1. **Loading the MNIST Dataset**: The MNIST dataset is loaded and split into training and testing sets.")
    st.write("2. **Creating the CNN Model**: A Convolutional Neural Network (CNN) is created with user-defined hyperparameters.")
    st.write("3. **Training the Model**: The model is trained with the specified hyperparameters, such as learning rate, dropout rate, and hidden units.")
    
    # Hyperparameters input
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
    dropout = st.sidebar.number_input("Dropout Rate", 0.0, 0.9, 0.5, 0.1)
    hidden_units = st.sidebar.number_input("Hidden Units", 10, 500, 100, 10)
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD"])
    epochs = st.sidebar.number_input("Number of Epochs",1,10,3,1)

    # Allow the user to input the batch size
    batch_size = st.sidebar.number_input("Batch Size", 1, 512, 64, 1)
    test_split = st.sidebar.number_input("Test data size",0.01,0.5,0.25,0.01)

    X_train, X_test, y_train, y_test  = load_mnist_data(batch_size, test_split)  # Pass the batch size to the data loader
    
    optimizer = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD

    # Train the model
    if st.button("Train Model"):
        st.write(f"Training the model with learning rate: {learning_rate}, dropout: {dropout}, hidden units: {hidden_units}, batch size: {batch_size}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_cnn_model(learning_rate, dropout, device, hidden_units, optimizer, epochs)

        # Convert target variable to numerical format
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Convert input data to the desired data type
        X_train = X_train.astype(np.float32)

        # Training results
        st.subheader("Training Results")

        # Add visual feedback during training
        with st.spinner("Training in progress..."):
            # Train for more epochs for better convergence
            for epoch in range(epochs):
                st.write(model.fit(X_train, y_train_encoded))
                
        if st.button("Save Model"):    
            model.save_params(
                f_params='digit_model_params.pkl',
                f_optimizer='digit_optimizer_state.pkl',
                f_history='digit_training_history.json'
            )
            st.success("Model saved successfully!")
        
        st.subheader("Model Evaluation")
        st.write("Evaluating the model...")
        
        X_test = X_test.astype(np.float32)
        y_test = label_encoder.fit_transform(y_test)
        
        # Evaluate the model on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Evaluation")
        # Test Accuracy
        st.subheader("Test Accuracy")
        st.write(
            "Test accuracy represents the percentage of correctly predicted instances in the test set. "
            "It is calculated as the ratio of correct predictions to the total number of instances in the test set."
        )
        st.write(f"Test Accuracy: {accuracy}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        st.write(f"- A confusion matrix is a table that describes the performance of a classification model.")
        st.text(f"- Rows: Actual classes\nColumns: Predicted classes")
        st.text(f"- Diagonal elements (from top-left to bottom-right): Correct predictions")
        st.text(f"- Off-diagonal elements: Misclassifications")
        st.text("It helps analyze how well the model performs for each class.")
        st.dataframe(conf_matrix, width=800, height=400)

        # Visualize Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5)
        st.subheader("Confusion Matrix Heatmap")
        st.pyplot(plt)

        # Classification Report
        class_report = classification_report(y_test, y_pred)
        st.subheader("Classification Report")
        st.text("A classification report provides precision, recall, and F1-score for each class.")
        st.text(f"- **Precision:** Of the instances predicted as positive, how many are actually positive?")
        st.text(f"- **Recall:** Of all the actual positive instances, how many did we predict as positive?")
        st.text(f"- **F1-Score:** The harmonic mean of precision and recall.")
        st.text(f"- **Support:** The number of true instances for each class.")
        st.text(class_report)
