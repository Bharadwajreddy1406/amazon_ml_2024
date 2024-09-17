# import pandas as pd
# from src.utils import download_images
#
# if __name__ == '__main__':
#     # Read the CSV file
#     train_df = pd.read_csv(r"train.csv")
#
#     # Extract the image links
#     image_links = train_df['image_link'].tolist()
#     image_links = image_links[:24]
#
#
#     # Specify the download folder
#     download_folder = 'downloads'
#
#     # Call the download_images function
#     download_images(image_links, download_folder, allow_multiprocessing=False)
#
import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import matplotlib.pyplot as plt


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def preprocess_image(image, target_size=(1200, 900), contrast_factor=1.5):
    open_cv_image = np.array(image)

    # Resize the image
    resized_image = cv2.resize(open_cv_image, target_size, interpolation=cv2.INTER_LINEAR)

    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(contrast_factor)
    open_cv_image = np.array(enhanced_image)

    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)

    binary_image = cv2.adaptiveThreshold(clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    blurred_image = cv2.medianBlur(binary_image, 5)

    # Edge detection
    edges = cv2.Canny(blurred_image, 100, 200)

    # Convert the preprocessed image back to PIL
    preprocessed_pil_image = Image.fromarray(edges)

    return preprocessed_pil_image, enhanced_image


def display_comparison(original_image, preprocessed_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original Image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Preprocessed Image
    axes[1].imshow(preprocessed_image, cmap='gray')
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


image_url = 'https://m.media-amazon.com/images/I/31EvJszFVfL.jpg'  # Replace wit image URL
original_image = fetch_image(image_url)
preprocessed_image, enhanced_image = preprocess_image(original_image)
display_comparison(original_image, enhanced_image)