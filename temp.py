import random
import pandas as pd
import torch
import easyocr
import re
from pathlib import Path
from src.constants import allowed_units
import gc
import matplotlib.pyplot as plt
import cv2

# Define the mapping for converting abbreviations to full units
unit_conversion_map = {
    'cm': 'centimetre',
    'ft': 'foot',
    'in': 'inch',
    'm': 'metre',
    'mm': 'millimetre',
    'yd': 'yard',
    'g': 'gram',
    'kg': 'kilogram',
    'ug': 'microgram',
    'mg': 'milligram',
    'oz': 'ounce',
    'lb': 'pound',
    'ton': 'ton',
    'kv': 'kilovolt',
    'mv': 'millivolt',
    'v': 'volt',
    'w': 'watt',
    'kw': 'kilowatt',
    'cl': 'centilitre',
    'cu_ft': 'cubic foot',
    'cu_in': 'cubic inch',
    'cup': 'cup',
    'dl': 'decilitre',
    'fl_oz': 'fluid ounce',
    'gal': 'gallon',
    'imp_gal': 'imperial gallon',
    'l': 'litre',
    'ul': 'microlitre',
    'ml': 'millilitre',
    'pt': 'pint',
    'qt': 'quart',
    'h': 'hour'
}

# Function to extract text from the image
def extract_text_from_image(image_path, use_cuda=False):
    reader = easyocr.Reader(['en'], gpu=use_cuda)
    result = reader.readtext(image_path)
    return result

# Function to clean and convert the extracted text into a standardized format
# Function to clean and convert the extracted text into a standardized format

def clean_extracted_text(extracted_text):
    cleaned_data = []
    # Regular expression to match various patterns including shorthand notations
    single_number_unit_pattern = r'(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h|CM|FT|IN|MM|M|YD|G|KG|UG|MG|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|H)'
    range_pattern = r'(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h|CM|FT|IN|MM|M|YD|G|KG|UG|MG|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|H)\s*to\s*(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h|CM|FT|IN|MM|M|YD|G|KG|UG|MG|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|H)'
    multiple_numbers_pattern = r'((\d+(\.\d+)?|\d+,\d+)(,\s*\d+(\.\d+)?|\d+,\d+)*?)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h|CM|FT|IN|MM|M|YD|G|KG|UG|MG|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|H)'
    bracketed_range_pattern = r'\[\s*(\d+(\.\d+)?|\d+,\d+)\s*,\s*(\d+(\.\d+)?|\d+,\d+)\s*\]\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h|CM|FT|IN|MM|M|YD|G|KG|UG|MG|OZ|LB|TON|KV|MV|V|W|KW|CL|CU_FT|CU_IN|CUP|DL|FL_OZ|GAL|IMP_GAL|L|UL|ML|PT|QT|H)'

    for text in extracted_text:
        # Match patterns like "10 kilogram to 15 kilogram"
        match = re.match(range_pattern, text[1])
        if match:
            cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(3)))
            cleaned_data.append((float(match.group(4).replace(',', '.')), match.group(6)))
        else:
            # Match single number and unit (like 10kg)
            match = re.match(single_number_unit_pattern, text[1])
            if match:
                cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(3)))
            else:
                # Match multiple numbers and unit (like 10, 20, 30kg)
                match = re.match(multiple_numbers_pattern, text[1])
                if match:
                    numbers = match.group(1).split(',')
                    for number in numbers:
                        cleaned_data.append((float(number.strip().replace(',', '.')), match.group(6)))
                else:
                    # Match bracketed range (like [10, 20]kg)
                    match = re.match(bracketed_range_pattern, text[1])
                    if match:
                        cleaned_data.append((float(match.group(1).replace(',', '.')), match.group(5)))
                        cleaned_data.append((float(match.group(3).replace(',', '.')), match.group(5)))
    return cleaned_data


# Function to map shorthand notations to original entity values
def map_units(cleaned_data):
    mapped_data = []
    for number, unit in cleaned_data:
        if unit in unit_conversion_map:
            mapped_unit = unit_conversion_map[unit]
            if mapped_unit in allowed_units:
                mapped_data.append((number, mapped_unit))
    return mapped_data

if __name__ == '__main__':
    # Specify the download folder
    download_folder = Path('downloads')

    # Get all image paths in the download folder
    image_paths = list(download_folder.glob('*.jpg'))
    image_path = random.choice(image_paths)

    print(image_paths)
    extracted_text = extract_text_from_image(str(image_path), use_cuda=True)
    print("Extracted Text:")
    print(extracted_text)
    cleaned_text = clean_extracted_text(extracted_text)
    mapped_text = map_units(cleaned_text)
    print("\n\nCleaned Text:\n", cleaned_text)
    print("\n\nMapped Text:\n", mapped_text)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Display the image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Print the extracted text
    print("Extracted Text:")
    for text in extracted_text:
        print(text[1])
        # print("Mapped ----> ", map_units([(text,)]))

    # Format the prediction
    if mapped_text:
        prediction = f"{mapped_text[0][0]} {mapped_text[0][1]}"
    else:
        prediction = ""