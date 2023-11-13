import streamlit as st
from front_page import about
from training import training_and_evaluation_app
from classify import image_classification_app
from explore_dataset import explore_dataset

from preprocessing import image_preprocessing_app

def main():
    st.sidebar.title("Table of Contents")
    page_options = ["About", "Explore Dataset", "Preprocessing Techniques","Training and Evaluation", "Image Classification"]
    selected_page = st.sidebar.radio("Select Page", page_options)

    if selected_page == "About":
        about()
    elif selected_page == "Training and Evaluation":
        training_and_evaluation_app()
    elif selected_page == "Image Classification":
        image_classification_app()
    elif selected_page == "Explore Dataset":
        explore_dataset()
    elif selected_page=="Preprocessing Techniques":
        image_preprocessing_app()


    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<h6>Made with ❤️ by <a href="https://www.linkedin.com/in/abhishek-p-singh/">@aps19</a></h6>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
