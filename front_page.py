import streamlit as st
st.set_page_config(layout="wide")

# About page
def about():
    st.title("About")
    st.write("Welcome to the MNIST Neural Network Trainer!")
    st.write("This app is designed to teach you how to define and train a simple Neural Network using PyTorch and skorch with Scikit-Learn. Below, we'll walk you through the process step by step.")

    st.subheader("Step 1: Loading the MNIST Dataset")
    st.write("We start by loading the MNIST dataset. The MNIST dataset contains images of handwritten digits (0-9), making it a great choice for learning computer vision and deep learning.")
    st.write("You can learn more about the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and the transformations used in the [PyTorch documentation](https://pytorch.org/vision/stable/transforms.html).")
    st.code("""
    import torchvision.transforms as transforms
    from torchvision import datasets

    # Load the MNIST dataset
    mnist = datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), download=True)
    """)

    st.subheader("Step 2: Defining the Neural Network Model")
    st.write("Next, we define our Neural Network model using PyTorch. In this example, we'll create a simple Convolutional Neural Network (CNN).")
    st.write("The CNN architecture used in this example consists of two convolutional layers followed by two fully connected layers.")
    st.write("You can learn more about defining neural networks in PyTorch from the [PyTorch documentation](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).")
    st.code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Cnn(nn.Module):
        def __init__(self):
            super(Cnn, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1024, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1024)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Cnn()
    """)

    st.subheader("CNN Architecture Details")
    st.write("In the CNN model, we have:")
    st.write("- Two convolutional layers (conv1 and conv2) with 32 and 64 filters, respectively, both using ReLU activation.")
    st.write("- Max-pooling layers following each convolutional layer to downsample the feature maps.")
    st.write("- Two fully connected layers (fc1 and fc2) for classification.")
    st.write("The choice of this architecture is common for image classification tasks, and you can experiment with different architectures for your specific problem.")


    st.subheader("Step 3: Training the Neural Network")
    st.write("Now, it's time to train our Neural Network model. We'll use skorch, a library that integrates PyTorch with Scikit-Learn, to make training easier.")
    st.write("Let's dive deeper into the training process:")

    st.subheader("3.1 Define Hyperparameters")
    st.write("Before training, we need to set hyperparameters such as learning rate and the number of epochs.")
    st.code("""
    # Define hyperparameters
    learning_rate = 0.001
    max_epochs = 10
    """)

    st.subheader("3.2 Create a NeuralNetClassifier")
    st.write("This step involves defining the model, [optimizer](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [loss criterion](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#loss-function), and other training settings.")
    
    st.code("""
    from skorch import NeuralNetClassifier
    import torch.optim as optim

    # Create a NeuralNetClassifier
    net = NeuralNetClassifier(
        module=model,  # The model we defined in Step 2
        optimizer=optim.Adam,  # Optimizer (e.g., Adam)
        criterion=nn.CrossEntropyLoss,  # Loss criterion (cross-entropy for classification)
        lr=learning_rate,  # Learning rate
        max_epochs=max_epochs,  # Number of training epochs
        device='cuda' if torch.cuda.is available() else 'cpu',  # Device for training (GPU or CPU)
    )
    """)
    st.write("In this step, we create a `NeuralNetClassifier` using skorch. Here's what each component does:")
    st.write("1. `module=model`: This specifies the neural network model we defined in Step 2. In this example, it's the `Cnn` class.")
    st.write("2. `optimizer=optim.Adam`: The optimizer is responsible for adjusting the model's parameters during training to minimize the loss. In this case, we're using the Adam optimizer, which is a popular choice for deep learning tasks. You can learn more about optimizers in the [PyTorch optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).")
    st.write("3. `criterion=nn.CrossEntropyLoss`: The criterion (or loss function) is used to calculate the difference between the model's predictions and the actual labels. For classification tasks, cross-entropy loss is commonly used. It's a measure of how well the model's predictions match the ground truth labels. Learn more about loss functions in the [PyTorch neural networks tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#loss-function).")
    st.write("4. `lr=learning_rate`: The learning rate determines the size of the steps taken during optimization. It's a crucial hyperparameter to adjust for efficient training. A small learning rate may result in slow convergence, while a large one may lead to overshooting the optimal solution.")
    st.write("5. `max_epochs=max_epochs`: This specifies the maximum number of training epochs (iterations). One epoch represents one pass through the entire training dataset. Training can stop earlier if the model converges before reaching the maximum epochs.")
    st.write("6. `device='cuda' if torch.cuda.is available() else 'cpu'`: This sets the device for training, either GPU ('cuda') or CPU ('cpu'). Using a GPU can significantly speed up training for deep learning models. Learn more about GPU support in PyTorch in the [PyTorch documentation](https://pytorch.org/docs/stable/notes/cuda.html).")

    st.write("By creating a `NeuralNetClassifier` with these settings, you've prepared the model for training. The next step is to train the model, which is covered in the following section.")


    st.subheader("3.3 Train the Model")
    st.write("The model is trained using the training data. Here's what the `fit` function does:")

    st.write("The `fit` function trains the neural network using the provided training data (`X_train` and `y_train`). It iteratively updates the model's parameters to minimize the loss, which measures how well the model's predictions match the actual labels. This process continues for the specified number of epochs (in this case, 10) or until convergence. It uses the Adam optimizer and cross-entropy loss for training.")

    st.write("You can learn more about the `fit` function and its parameters in the [skorch documentation](https://skorch.readthedocs.io/en/stable/net.html#skorch.net.NeuralNet.fit).")

    st.code("""
    # Train the model
    net.fit(X_train, y_train)
    """)

    st.write("Congratulations! You have now learned how to set hyperparameters, create a `NeuralNetClassifier`, and train a neural network using skorch.")

    st.write("You can experiment with different hyperparameters and see how they affect the training process and model performance on the 'Training and Evaluation' page.")

    st.write("Happy learning!")
