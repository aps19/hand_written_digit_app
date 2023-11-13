import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

st.set_page_config(layout="wide")

# Function to download and save the dataset to a local repository
# Function to download and save the dataset to a local repository
def download_and_save_dataset():
    st.write("Downloading the MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    st.write("Dataset downloaded successfully!")

    # Convert data and target to numpy arrays
    data_array = np.array(mnist.data)
    target_array = np.array(mnist.target)

    # Combine data and target arrays into a dictionary
    dataset_dict = {'data': data_array, 'target': target_array}

    # Save the dataset as an npz file
    np.savez('mnist_dataset.npz', **dataset_dict)

    st.write("Dataset saved as 'mnist_dataset.npz'.")
    
def plot_mnist_collage(images, labels, rows=10, cols=10, figsize=(10, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            ax = axes[i, j]
            ax.imshow(images[index].reshape(28, 28), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Label: {labels[index]}", fontsize=8)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()

    # Display the matplotlib figure in Streamlit
    st.pyplot(fig)

def download_dataset(data, target):
    # Convert data and target to numpy arrays
    data_array = np.array(data)
    target_array = np.array(target)

    # Combine data and target arrays into a dictionary
    dataset_dict = {'data': data_array, 'target': target_array}

    # Save the dataset as a numpy file
    np.savez('mnist_dataset.npz', **dataset_dict)

def plot_interactive_images(images, labels, cols=3, rows=3, image_size=(100, 100)):
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            image = images[index].reshape(28, 28)

            # Convert image to uint8 for proper display
            image = (image * 255).astype(np.uint8)

            # Resize the image for better display
            image = np.array(Image.fromarray(image).resize(image_size))

            # Display the image with label
            st.image(image, caption=f"Label: {labels[index]}", use_column_width=False)

from sklearn.metrics import pairwise_distances


def apply_clustering(X, y, n_clusters=10):
    st.subheader("Clustering: K-Means")
    st.write(f"Apply K-Means clustering to group similar images together.")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_features)

    # Calculate pairwise distances between cluster centers
    cluster_centers = kmeans.cluster_centers_
    center_distances = pairwise_distances(cluster_centers)

    # Create a figure for the clustered data
    fig, ax = plt.subplots()

    # Scatter plot for each cluster
    for cluster_label in range(n_clusters):
        cluster_points = reduced_features[kmeans.labels_ == cluster_label]
        # Get the majority class in the cluster
        majority_class = pd.Series(y[kmeans.labels_ == cluster_label]).value_counts().idxmax()
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            label=f"Class {majority_class}"
        )

    # Add labels and title
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"K-Means Clustering (K={n_clusters})")

    # Add a legend
    ax.legend()

    # Display the Matplotlib figure using Streamlit
    st.pyplot(fig)

    st.write(
        "K-Means clustering is a technique used to group similar data points together. In the context of the MNIST dataset, "
        "K-Means is applied to group similar images based on their features. Here's what you can learn from the K-Means clustering graph:"
    )
    st.markdown(
        "- **Cluster Representation:** Each cluster in the graph represents a group of images that share similarities in their features, as determined by K-Means."
    )
    st.markdown(
        "- **Visualization of Similarity:** Images within the same cluster are visually similar, as they are closer together in the graph."
    )
    st.markdown(
        "- **Principal Components:** The x and y-axis represent the first two principal components obtained from PCA, capturing the most significant variations in the data."
    )

    st.write(
        "Understanding the distribution of images in clusters can provide insights into the natural groupings of handwritten digits. "
        "This information can be valuable for further analysis or for improving the performance of machine learning models."
    )

    st.subheader("Pairwise Distances between Cluster Centers")
    st.write(
        "The matrix below represents the pairwise distances between the centers of the clusters. "
        "It provides an indication of how well-separated the clusters are in the feature space."
    )
    st.write(center_distances)



# Function to load the dataset and perform exploration
def explore_dataset():
    
    st.title("Explore Your Dataset")
    st.write("Welcome to the interactive dataset exploration page! Let's learn more about the MNIST dataset.")
    data = np.load('mnist_dataset.npz', allow_pickle=True)
    X, y = data['data'], data['target']
    
    # Sidebar options
    st.sidebar.header("Options")
    dataset_overview = st.sidebar.checkbox("Dataset Overview", help="Check this box to view overview of the dataset.")
    check_images = st.sidebar.checkbox("View Example Images", help="Check this box to view images of each class.")
    digit_distribution = st.sidebar.checkbox("Digit Distribution", help="Check this box to view the Distribution of classes in MNIST dataset.")
    descriptive_stats = st.sidebar.checkbox("Descriptive Statistics", help="Check this box to view the descriptive statistics.")
    apply_kmeans = st.sidebar.checkbox("Apply K-Means Clustering", help="Check this box to apply K-Means clustering.")
    # Button to download and save the dataset
    if st.sidebar.button("Download and Save Dataset", help="Click to download and save the MNIST dataset as a numpy array."):
        download_and_save_dataset()
        
    
    # Create two columns layout
    left_column, right_column = st.columns(2)

    if dataset_overview:
        # Display basic information about the dataset in the right column
       with left_column:
        st.subheader("Dataset Overview")
        st.write(f"- The MNIST dataset comprises {len(X)} images of handwritten digits.")
        st.write(f"- Each image is a 28x28 pixel grayscale representation of a handwritten digit ranging from 0 to 9.")
        st.write(f"- The dataset encompasses {X.shape[1]} features, where each feature corresponds to a pixel value.")
        st.write(f"- Pixel values in the images range from 0 to 255, with 0 representing white and 255 representing black.")

        # Additional information about the nature of the dataset
        st.write("- Widely used in machine learning, particularly in computer vision, the MNIST dataset serves as a benchmark for evaluating the performance of various image processing and classification algorithms.")
        st.write("- The primary objective is to accurately classify handwritten digits based on their pixel values.")

        # Display example images in the left column
        with right_column:
            st.subheader("Example Images from the MNIST Dataset")
            st.write("Here, you can see a collage of example images from the MNIST dataset along with their corresponding labels.")
            plot_mnist_collage(X, y)

    if check_images:
        st.subheader("Example images from a single class")
        left_column2, right_column2 = st.columns(2)    
            
        with left_column2:
            # Allow the user to choose the class
            selected_class = st.selectbox("Choose a class label", sorted(np.unique(y)))

            # Select random images from the chosen class
            random_image_indices = np.where(y == selected_class)[0]
            random_image_indices = np.random.choice(random_image_indices, size=25, replace=False)
            random_images = X[random_image_indices]
            random_labels = y[random_image_indices]
        
        with right_column2:
            # Plot the interactive grid for the selected class
            plot_mnist_collage(random_images, random_labels,rows=3, cols=3)
            
    if digit_distribution:
        # Visualize the distribution of digits
        st.subheader("Digit Distribution")
        digit_counts = pd.Series(y).value_counts().sort_index()
        fig_digit_dist = plt.figure(figsize=(8, 5))
        sns.barplot(x=digit_counts.index, y=digit_counts.values, palette="viridis")
        plt.title("Distribution of Digits in MNIST Dataset")
        plt.xlabel("Digit")
        plt.ylabel("Count")
        st.pyplot(fig_digit_dist)
        
        st.write(
            "Understanding the distribution of digits in the dataset is crucial for building a successful model. "
            "Here's what you can learn from the digit distribution:"
        )
        st.markdown(
            "- **Balanced Dataset:** A balanced dataset has roughly equal samples for each digit. This ensures that the "
            "model has sufficient data for each digit, preventing biases toward certain digits."
        )
        st.markdown(
            "- **Imbalanced Dataset:** If certain digits have significantly fewer samples than others, the model may "
            "struggle to learn those digits well. It's important to be aware of any imbalance and address it if needed."
        )
        st.write(
            "In the bar plot above, each bar represents a digit (0 through 9), and the height of the bar indicates "
            "the count of occurrences for that digit in the dataset."
        )
        st.write(
            "For example, the bar for digit '1' is tallest, it means there are more images of the digit '1' in the dataset. "
            "Conversely, the bar for digit '5' is shortest, this means that there are fewer samples of digit '5'."
        )
        st.write(
            "Take a moment to analyze the digit distribution. Is the dataset balanced? Are there any digits with notably "
            "fewer samples? Understanding these aspects will guide your approach to training and evaluating your model."
        )
        
        # Print the number of samples for each class
        st.subheader("Number of Samples for Each Class")
        st.write("Here's the count of samples for each digit class:")
        st.table(pd.DataFrame({"Digit": digit_counts.index, "Count": digit_counts.values}))

        st.write(
            "Our dataset appears to be relatively balanced as the count of samples for each digit class is comparable. "
            "In a balanced dataset, each class has roughly the same number of samples, or the differences in sample counts are not extreme. "
            "In this case, the counts for digits 0 through 9 are fairly close, indicating a balanced distribution."
        )
        
    if descriptive_stats:
        # Descriptive statistics about pixel values
        st.subheader("Descriptive Statistics")

        # Display mean, median, and standard deviation of pixel values
        mean_pixel_value = np.mean(X)
        median_pixel_value = np.median(X)
        std_pixel_value = np.std(X)

        # Display the calculated statistics
        st.write(f"Mean Pixel Value: {mean_pixel_value:.2f}")
        st.write(f"Median Pixel Value: {median_pixel_value:.2f}")
        st.write(f"Standard Deviation of Pixel Values: {std_pixel_value:.2f}")

        
        # Display mean, median, and standard deviation with explanations
        st.write(
            "Descriptive statistics provide insights into the overall characteristics of pixel values in the dataset. "
            "Here are some key statistics:"
        )
        # Explain mean pixel value
        st.write(
            f"- **Mean Pixel Value:** The average darkness of all pixels. A mean value of 33.39 indicates that, "
            "on average, pixels are relatively dark."
        )
        st.write(
            f"- **Median Pixel Value:** The middle pixel value when all values are sorted. It gives a sense of the central tendency "
            "and is less affected by extreme values than the mean. In this case, the median value is 0.00, suggesting that there are "
            "most pixels with very low values."
        )
        st.write(
            f"- **Standard Deviation:** A measure of the amount of variation or dispersion in pixel values. A higher standard "
            "deviation (78.65 in this case) indicates more variability in pixel darkness across the images. This suggests that there "
            "is a wide range of pixel values, contributing to the diversity in the dataset."
        )
        
        st.write(
            "Interpretation of these statistics can help you understand the overall intensity and variability of pixel values "
            "in the images. It's valuable information for preprocessing and choosing appropriate normalization techniques."
        )
        
        st.write(
            "For instance, the relatively high standard deviation indicates that pixel values exhibit significant variability, suggesting that normalization techniques might be beneficial for enhancing model performance. "
            "Additionally, the median pixel value being 0.00 indicates the presence of pixels with very low values, which could impact the model's ability to discern features in those regions."
        )

    
    if apply_kmeans:
        # Apply K-Means clustering
        apply_clustering(X,y)
    
    
