import pandas as pd
import torch
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
    # Specify the download folder
    download_folder = Path('downloads')

    # Get all image paths in the download folder
    image_paths = list(download_folder.glob('*.jpg'))

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()

    # Initialize a list to store results
    results = []

    # Extract and clean text from downloaded images
    for index, image_path in enumerate(image_paths):
        extracted_text = extract_text_from_image(str(image_path), use_cuda=use_cuda)
        cleaned_text = clean_extracted_text(extracted_text)

        # Format the prediction
        if cleaned_text:
            prediction = f"{cleaned_text[0][0]} {cleaned_text[0][1]}"
        else:
            prediction = ""

        # Append results to the list
        results.append({
            'index': index + 1,
            'prediction': prediction
        })

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/test_out.csv', index=False)