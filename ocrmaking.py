import pandas as pd
import torch

from src.utils import download_images
import easyocr
import re
from pathlib import Path

def extract_text_from_image(image_path, use_cuda=False):
    reader = easyocr.Reader(['en'], gpu=use_cuda)
    result = reader.readtext(image_path)
    return result

def clean_extracted_text(extracted_text):
    cleaned_data = []
    for text in extracted_text:
        match = re.match(r'(\d+(\.\d+)?)\s*([a-zA-Z]+)', text[1])
        if match:
            cleaned_data.append((float(match.group(1)), match.group(3)))
    return cleaned_data

if __name__ == '__main__':
    # Read the CSV file
    train_df = pd.read_csv(r"dataset/train.csv")

    # Extract the image links
    image_links = train_df['image_link'].tolist()
    image_links = image_links[:10]

    # Specify the download folder
    download_folder = 'downloads'

    # Call the download_images function
    # download_images(image_links, download_folder, allow_multiprocessing=False)

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()

    # Initialize a list to store results
    results = []

    # Extract and clean text from downloaded images
    for image_link in image_links:
        image_path = Path(download_folder) / Path(image_link).name
        extracted_text = extract_text_from_image(str(image_path), use_cuda=use_cuda)
        cleaned_text = clean_extracted_text(extracted_text)
        print(f"Extracted text from {image_path}: {cleaned_text}")

        # Append results to the list
        results.append({
            'image_path': str(image_path),
            'extracted_text': cleaned_text
        })

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/extracted_text.csv', index=False)