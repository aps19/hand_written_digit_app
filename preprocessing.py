import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from skimage import exposure, color
from skimage.feature import greycomatrix
from skimage.measure import shannon_entropy
from skimage.feature import greycoprops

from skimage.color import rgb2gray

from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import random
import matplotlib.pyplot as plt
import cv2
import io

def load_data():
    # Load MNIST dataset
    data = np.load('mnist_dataset.npz', allow_pickle=True)
    X, y = data['data'], data['target']

    # Preprocess the datas
    # X /= 255.0

    return X, y

# Function to apply rotation to the image
def rotate_image(image, angle):
    return image.rotate(angle)

# Function to apply flipping to the image
def flip_image(image, horizontal=False, vertical=False):
    if horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

# Function to adjust brightness and contrast of the image
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(factor)
    return adjusted_image

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(factor)
    return adjusted_image

# Function to apply random cropping to the image
def random_crop(image, crop_size):
    width, height = image.size
    left = np.random.randint(0, width - crop_size)
    top = np.random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))

# Function to apply Gaussian blurring to the image
def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

# Function to apply histogram equalization to the image
def apply_histogram_equalization(image):
    img_array = np.array(image)
    img_eq = exposure.equalize_hist(img_array)
    img_eq = (img_eq * 255).astype(np.uint8)
    return Image.fromarray(img_eq)

# Function to apply color jittering to the image
def apply_color_jitter(image, jitter_factor):
    img_array = np.array(image)
    img_jittered = img_array + jitter_factor * np.random.randn(*img_array.shape)
    img_jittered = np.clip(img_jittered, 0, 255).astype(np.uint8)
    return Image.fromarray(img_jittered)


# Function to apply normalization to the image
def apply_normalization(image, mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_tensor = transform(image)
    return transforms.ToPILImage()(img_tensor)

# Function to rescale intensity values in the image
def rescale_intensity(image, min_val, max_val):
    img_array = np.array(image)
    img_rescaled = exposure.rescale_intensity(img_array, in_range=(0, 255), out_range=(min_val, max_val))
    img_rescaled = (img_rescaled * 255).astype(np.uint8)
    return Image.fromarray(img_rescaled)

# Function to extract GLCM features from the image
def extract_glcm_features(image):
    # Convert the image to grayscale
    gray_image = rgb2gray(np.array(image))

    # Convert the grayscale image to unsigned integer type
    gray_image = (gray_image * 255).astype(np.uint8)

    # Compute GLCM features
    glcm = greycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, prop='contrast')[0, 0]
    dissimilarity = greycoprops(glcm, prop='dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, prop='homogeneity')[0, 0]
    energy = greycoprops(glcm, prop='energy')[0, 0]
    correlation = greycoprops(glcm, prop='correlation')[0, 0]

    return contrast, dissimilarity, homogeneity, energy, correlation

# Function to apply edge detection to the image
def apply_edge_detection(image):
    img_array = np.array(image)
    edges = cv2.Canny(img_array, 100, 200)
    return Image.fromarray(edges)

# Function to shift histogram of the image
def shift_histogram(image, shift_factor):
    img_array = np.array(image)
    img_shifted = exposure.adjust_gamma(img_array, gamma=shift_factor)
    img_shifted = (img_shifted * 255).astype(np.uint8)
    return Image.fromarray(img_shifted)

def apply_processing_techniques(original_image, processing_options):
    image = original_image.copy()

    # Apply user-selected preprocessing techniques
    if processing_options["rotate"]:
        image = rotate_image(image, processing_options["rotate_value"])
    if processing_options["flip_horizontal"]:
        image = flip_image(image, True, False)
    if processing_options["flip_vertical"]:
        image = flip_image(image, False, True)
    if processing_options["adjust_brightness"]:
        image = adjust_brightness(image, processing_options["adjust_brightness_value"])
    if processing_options["adjust_contrast"]:
        image = adjust_contrast(image, processing_options["adjust_contrast_value"])
    if processing_options["apply_crop"]:
        image = random_crop(image, processing_options["crop_size"])
    if processing_options["apply_gaussian_blur"]:
        image = apply_gaussian_blur(image, processing_options["gaussian_blur_radius"])
    if processing_options["apply_histogram_equalization"]:
        image = apply_histogram_equalization(image)
    if processing_options["apply_color_jitter"]:
        image = apply_color_jitter(image, processing_options["color_jitter_factor"])
    if processing_options["apply_normalization"]:
        image = apply_normalization(image, (processing_options["normalization_mean"],), (processing_options["normalization_std"],))
    if processing_options["rescale_intensity"]:
        image = rescale_intensity(image, processing_options["rescale_intensity_min"], processing_options["rescale_intensity_max"])
    if processing_options["shift_histogram"]:
        image = shift_histogram(image, processing_options["shift_histogram_factor"])

    return image

def image_preprocessing_app():
    st.title("Image Preprocessing Techniques")
    st.write("Choose an option to upload an image or select a random image from the dataset.")

    option = st.radio("Choose an option:", ("Upload an Image", "Random Image from Dataset"))

    uploaded_image = None

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Convert the uploaded image to a NumPy array
            uploaded_image = np.array(Image.open(uploaded_file).convert('L')) / 255.0  # Convert to grayscale and normalize
            # Convert the grayscale image to RGB format for display
            uploaded_image_rgb = np.stack((uploaded_image,) * 3, axis=-1)
    else:
        st.info("Selecting a random image from the dataset.")
        X, _ = load_data()
        random_index = random.randint(0, len(X) - 1)
        uploaded_image = X[random_index].reshape(28,28)
        # Convert the grayscale image to RGB format for display
        uploaded_image_rgb = np.stack((uploaded_image,) * 3, axis=-1)


    if uploaded_image is not None:
        # Display the original and processed images in a two-column layout
        col1, col2 = st.columns(2)

        # Display the original image in the left column
        col1.header("Original Image")
        original_image = Image.fromarray((uploaded_image_rgb * 255).astype(np.uint8))  # Convert to 8-bit for display
        col1.image(original_image, caption="Original Image", use_column_width=False)
        
        # Use st.sidebar for user options
        with st.sidebar:
            st.subheader("User Options")

            checkbox_keys = {
                "rotate": "Rotate",
                "flip_horizontal": "Flip Horizontal",
                "flip_vertical": "Flip Vertical",
                "adjust_brightness": "Adjust Brightness",
                "adjust_contrast": "Adjust Contrast",
                "apply_crop": "Apply Random Crop",
                "apply_gaussian_blur": "Apply Gaussian Blur",
                "apply_histogram_equalization": "Apply Histogram Equalization",
                "apply_color_jitter": "Apply Color Jitter",
                "apply_normalization": "Apply Normalization",
                "rescale_intensity": "Rescale Intensity",
                "shift_histogram": "Shift Histogram",
                "show_glcm_features": "Show GLCM Features",
                "show_edge_detection": "Show Edge Detection",
            }

            processing_options = {}

            for key, label in checkbox_keys.items():
                checkbox_value = st.checkbox(label, key=f"{key}_checkbox", value=False)
                processing_options[key] = checkbox_value

                if checkbox_value:
                    if key in ["rotate", "adjust_brightness", "adjust_contrast"]:
                        processing_options[f"{key}_value"] = st.slider(
                            f"{label} Value",
                            min_value=0.0,
                            max_value=360.0 if key == "rotate" else 2.0,
                            value=1.0,
                            step=0.1,
                        )
                    elif key == "apply_crop":
                        processing_options["crop_size"] = st.slider(
                            "Random Crop Size",
                            min_value=1,
                            max_value=min(original_image.size)-3,  # Adjust the max value accordingly
                            value=min(original_image.size) // 2,  # Adjust the default value accordingly
                            step=1,
                        )
                    elif key == "apply_gaussian_blur":
                        processing_options["gaussian_blur_radius"] = st.slider(
                            "Gaussian Blur Radius",
                            min_value=0,
                            max_value=10,
                            value=0,
                            step=1,
                        )
                    elif key == "apply_color_jitter":
                        processing_options["color_jitter_factor"] = st.slider(
                            "Color Jittering Factor",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1,
                        )
                    elif key == "apply_normalization":
                        processing_options["normalization_mean"] = st.slider(
                            "Normalization Mean",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01,
                        )
                        processing_options["normalization_std"] = st.slider(
                            "Normalization Standard Deviation",
                            min_value=0.01,
                            max_value=2.0,
                            value=1.0,
                            step=0.01,
                        )
                    elif key == "rescale_intensity":
                        processing_options["rescale_intensity_min"] = st.slider(
                            "Rescale Intensity Min",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                        )
                        processing_options["rescale_intensity_max"] = st.slider(
                            "Rescale Intensity Max",
                            min_value=1.0,
                            max_value=2.0,
                            value=1.0,
                            step=0.01,
                        )
                    elif key == "shift_histogram":
                        processing_options["shift_histogram_factor"] = st.slider(
                            "Shift Histogram Factor",
                            min_value=0.01,
                            max_value=2.0,
                            value=1.0,
                            step=0.01,
                        )

        # Apply user-selected preprocessing techniques
        processed_image = apply_processing_techniques(original_image, processing_options)

        # Display the processed image in the right column
        col2.header("Processed Image")
        col2.image(processed_image, caption="Processed Image", use_column_width=False)

        # Provide additional analysis or display if needed
        if processing_options["show_glcm_features"]:
            glcm_features = extract_glcm_features(processed_image)
            st.subheader("GLCM Features")
            st.write(f"Contrast: {glcm_features[0]}")
            st.write(f"Dissimilarity: {glcm_features[1]}")
            st.write(f"Homogeneity: {glcm_features[2]}")
            st.write(f"Energy: {glcm_features[3]}")
            st.write(f"Correlation: {glcm_features[4]}")

        if processing_options["show_edge_detection"]:
            edge_detected_image = apply_edge_detection(processed_image)
            st.subheader("Edge Detection")
            st.image(edge_detected_image, caption="Edge Detection", use_column_width=True)
