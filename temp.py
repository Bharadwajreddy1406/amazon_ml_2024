import random
import pandas as pd
import torch
import easyocr
import re
from pathlib import Path
from src.constants import allowed_units
import gc

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import matplotlib.pyplot as plt


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


def extract_text_from_image(image_path, use_cuda=False):
    reader = easyocr.Reader(['en'], gpu=use_cuda)
    result = reader.readtext(image_path)
    return result


def clean_extracted_text(extracted_text):
    cleaned_data = []
    # Patterns

    single_number_unit_pattern = r'.*?(\d+(\.\d+)?|\d+,\d+)\s*(CM|FT|IN|MM|MG|KG|UG|MG|G|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|YD|H|cm|ft|in|mm|mg|kg|ug|g|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|yd|h).*?'
    range_pattern = r'(\d+(\.\d+)?|\d+,\d+)\s*(CM|FT|IN|MM|MG|KG|UG|MG|G|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|YD|H|cm|ft|in|mm|mg|kg|ug|g|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|yd|h)\s*to\s*(\d+(\.\d+)?|\d+,\d+)\s*(CM|FT|IN|MM|MG|KG|UG|MG|G|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|YD|H|cm|ft|in|mm|mg|kg|ug|g|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|yd|h)'
    multiple_numbers_pattern = r'((\d+(\.\d+)?|\d+,\d+)(,\s*\d+(\.\d+)?|\d+,\d+)*?)\s*(CM|FT|IN|MM|MG|KG|UG|MG|G|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|YD|H|cm|ft|in|mm|mg|kg|ug|g|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|yd|h)'
    bracketed_range_pattern = r'\[\s*(\d+(\.\d+)?|\d+,\d+)\s*,\s*(\d+(\.\d+)?|\d+,\d+)\s*\]\s*(CM|FT|IN|MM|MG|KG|UG|MG|G|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|YD|H|cm|ft|in|mm|mg|kg|ug|g|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|yd|h)'

    for text in extracted_text:
        match = re.match(range_pattern, text[1])
        if match:
            cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(3)))
            cleaned_data.append((float(match.group(4).replace(',', '.')), match.group(6)))
        else:
            match = re.match(single_number_unit_pattern, text[1])
            if match:
                cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(3)))
            else:
                match = re.match(multiple_numbers_pattern, text[1])
                if match:
                    numbers = match.group(1).split(',')
                    for number in numbers:
                        cleaned_data.append((float(number.strip().replace(',', '.')), match.group(6)))
                else:
                    match = re.match(bracketed_range_pattern, text[1])
                    if match:
                        cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(5)))
                        cleaned_data.append((float(match.group(3).replace(',', '.')), match.group(5)))
    return cleaned_data


def map_units(cleaned_data):
    unit_conversion_map = {
        'cm': 'centimetre',
        'CM': 'centimetre',
        'ft': 'foot',
        'FT': 'foot',
        'in': 'inch',
        'IN': 'inch',
        'm': 'metre',
        'M': 'metre',
        'mm': 'millimetre',
        'MM': 'millimetre',
        'yd': 'yard',
        'YD': 'yard',
        'g': 'gram',
        'G': 'gram',
        'kg': 'kilogram',
        'KG': 'kilogram',
        'ug': 'microgram',
        'UG': 'microgram',
        'mg': 'milligram',
        'MG': 'milligram',
        'oz': 'ounce',
        'OZ': 'ounce',
        'lb': 'pound',
        'LB': 'pound',
        'ton': 'ton',
        'TON': 'ton',
        'kv': 'kilovolt',
        'KV': 'kilovolt',
        'mv': 'millivolt',
        'MV': 'millivolt',
        'v': 'volt',
        'V': 'volt',
        'w': 'watt',
        'W': 'watt',
        'kw': 'kilowatt',
        'KW': 'kilowatt',
        'cl': 'centilitre',
        'CL': 'centilitre',
        'cu_ft': 'cubic foot',
        'CU_FT': 'cubic foot',
        'cu_in': 'cubic inch',
        'CU_IN': 'cubic inch',
        'cup': 'cup',
        'CUP': 'cup',
        'dl': 'decilitre',
        'DL': 'decilitre',
        'fl_oz': 'fluid ounce',
        'FL_OZ': 'fluid ounce',
        'gal': 'gallon',
        'GAL': 'gallon',
        'imp_gal': 'imperial gallon',
        'IMP_GAL': 'imperial gallon',
        'l': 'litre',
        'L': 'litre',
        'ul': 'microlitre',
        'UL': 'microlitre',
        'ml': 'millilitre',
        'ML': 'millilitre',
        'pt': 'pint',
        'PT': 'pint',
        'qt': 'quart',
        'QT': 'quart',
        'h': 'hour',
        'H': 'hour'
    }
    allowed_units = set(unit_conversion_map.values())
    mapped_data = []
    for number, unit in cleaned_data:
        if unit in unit_conversion_map:
            mapped_unit = unit_conversion_map[unit]
            if mapped_unit in allowed_units:
                mapped_data.append((number, mapped_unit))
    return mapped_data


if __name__ == '__main__':
    download_folder = Path('downloads')

    # Get all image paths in the download folder
    image_paths = list(download_folder.glob('*.jpg'))
    # image_path = random.choice(image_paths)
    image_path = r"C:\Users\reddy\PycharmProjects\amazon_ml\final_custos\downloads\81S2+GnYpTL.jpg"

    # Print the selected image path
    print(f"Selected Image Path: {image_path}")

    # Preprocess the image
    original_image = Image.open(image_path)
    preprocessed_image, enhanced_image = preprocess_image(original_image)

    # Save the enhanced image to a temporary file
    temp_image_path = 'temps/temp_enhanced_image.jpg'
    enhanced_image = enhanced_image.convert('RGB')

    enhanced_image.save(temp_image_path)

    # Extract text from the enhanced image
    extracted_text = extract_text_from_image(temp_image_path, use_cuda=True)
    print("Extracted Text:")
    print(extracted_text)
    cleaned_text = clean_extracted_text(extracted_text)
    mapped_text = map_units(cleaned_text)
    print("\n\nCleaned Text:\n", cleaned_text)
    print("\n\nMapped Text:\n", mapped_text)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Display the original and enhanced images
    display_comparison(original_image, enhanced_image)

    # Print the extracted text
    print("Extracted Text:")
    for text in extracted_text:
        print(text[1])

    # Format the prediction
    if mapped_text:
        prediction = f"{mapped_text[0][0]} {mapped_text[0][1]}"
    else:
        prediction = ""

gc.collect()