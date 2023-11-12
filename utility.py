# Importing Libraries
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

def load_data():
    # Load MNIST dataset
    data = np.load('mnist_dataset.npz', allow_pickle=True)
    X, y = data['data'], data['target']

    # Preprocess the data
    X /= 255.0

    return X, y

def load_model():
    # Load the model from the specified path
    model = create_cnn_model()
    model.initialize()  
    model.load_params(
            f_params='digit_model_params.pkl',
            f_optimizer='digit_optimizer_state.pkl',
            f_history='digit_training_history.json'
        )

    return model

def create_cnn_model(learning_rate=0.01, dropout=0.5, device='cpu', hidden_units=100, optimizer=torch.optim.Adam,epochs=10):
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