import streamlit as st
from skorch import NeuralNetClassifier
import random
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Import the first page from front_page.py
from front_page import about
from training import training_and_evaluation_app

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "About": about,
        "Training and Evaluation": training_and_evaluation_app,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made with ❤️ by <a href="https://www.linkedin.com/in/abhishek-p-singh/">@aps19</a></h6>',
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
