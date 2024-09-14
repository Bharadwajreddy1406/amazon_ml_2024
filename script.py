import easyocr
import pandas as pd
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import torch
import re

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

# Function to extract text from an image
def extract_text_from_image(image_path, use_cuda=False):
    reader = easyocr.Reader(['en'], gpu=use_cuda)
    result = reader.readtext(image_path)
    return result

# Function to clean and convert the extracted text into a standardized format
def clean_extracted_text(extracted_text):
    cleaned_data = []
    single_number_unit_pattern = r'(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h)'
    range_pattern = r'(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h)\s*to\s*(\d+(\.\d+)?|\d+,\d+)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h)'
    multiple_numbers_pattern = r'((\d+(\.\d+)?|\d+,\d+)(,\s*\d+(\.\d+)?|\d+,\d+)*?)\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h)'
    bracketed_range_pattern = r'\[\s*(\d+(\.\d+)?|\d+,\d+)\s*,\s*(\d+(\.\d+)?|\d+,\d+)\s*\]\s*(cm|ft|in|mm|m|yd|g|kg|ug|mg|oz|lb|ton|kv|mv|v|w|kw|cl|cu_ft|cu_in|cup|dl|fl_oz|gal|imp_gal|l|ul|ml|pt|qt|h)'

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

# Function to map shorthand notations to original entity values
def map_units(cleaned_data):
    allowed_units = set(unit_conversion_map.values())
    mapped_data = []
    for number, unit in cleaned_data:
        if unit in unit_conversion_map:
            mapped_unit = unit_conversion_map[unit]
            if mapped_unit in allowed_units:
                mapped_data.append((number, mapped_unit))
    return mapped_data





df = pd.read_csv(r"dataset/train1.csv")

df.rename(columns={"Unnamed: 0": "index"}, inplace=True)
df = df[["index", "entity_value", "image_link"]]

# Step 2: Extract image names from the DataFrame (from 'image_link')
df['image_name'] = df['image_link'].apply(lambda x: x.split("/")[-1])

# Step 3: Get the list of image paths from the folder
download_folder = Path('downloads')
image_paths = list(download_folder.glob('*.jpg'))

# Step 4: Extract the image names from the folder paths
folder_image_names = {str(image_path).split("\\")[-1]: str(image_path) for image_path in image_paths}

# Step 5: Map the DataFrame image names to the corresponding image paths
df['image_path'] = df['image_name'].map(folder_image_names)

# Step 6: Check the result
df = df[['image_name', 'image_path','entity_value']]


def process_images(df):
    extracted_data = []

    i = 0
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        try:
            if i == 20:
                return df

            image_path = row['image_path']
            if row["image_name"] in image_path:
                print("Image path found", row["image_name"], row["image_path"])

                if pd.notna(image_path):  # Check if image path exists
                    # Step 2: Perform OCR on the image and clean the text
                    extracted_text = extract_text_from_image(str(image_path), use_cuda=True)
                    # print(extracted_text)
                    cleaned_text = clean_extracted_text(extracted_text)
                    mapped_text = map_units(cleaned_text)

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Step 3: Append the cleaned/mapped text to the list
                    extracted_data.append(mapped_text)
            else:
                df.to_csv('outputs/finalTests.csv', index=True)
                return df
        except Exception as e:
            df.to_csv('outputs/finalTests.csv', index=True)
            return df

    # Step 4: Add the extracted data as a new column in the DataFrame
    df['extracted_text'] = extracted_data
    i += 1

    return df

# Step 5: Call the function and store the results
df = process_images(df)
# df.to_csv('outputs/finalTests.csv', index = True)
# print(df.head(10))