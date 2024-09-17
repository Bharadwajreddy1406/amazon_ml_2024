# import pandas as pd
#
# # Load your original and modified DataFrames
# original_df = pd.read_csv('test.csv')
# modified_df = pd.read_csv(r'C:\Users\reddy\PycharmProjects\amazon_ml\dataset\test.csv')  # After removing duplicates
#
# # Check for duplicates in the original DataFrame based on the key column (e.g., 'image_link')
# duplicates_in_original = original_df[original_df.duplicated(subset='image_link', keep=False)]
#
# # Now get the rows that are missing from the modified DataFrame
# # We assume 'image_link' is the key column used for removing duplicates
# missing_rows = original_df[~original_df.index.isin(modified_df.index)]
#
# # Check shape of missing rows
# print(f"Missing rows shape: {missing_rows.shape}")
#
# # Concatenate missing rows back to modified DataFrame
# new_df = pd.concat([modified_df, missing_rows])
#
# # Optionally, sort the DataFrame by index to maintain the original order
# new_df = new_df.sort_index()
#
# # Final check on the shape
# print(f"Original shape: {original_df.shape}, Modified shape: {modified_df.shape}, New shape after reinsertion: {new_df.shape}")
import pandas as pd
from src.utils import download_images

# Load the original and modified DataFrames
original_df = pd.read_csv(r"test.csv")
modified_df = pd.read_csv(r"C:\Users\reddy\PycharmProjects\amazon_ml\dataset\test.csv")  # After removing duplicates

# Find missing rows in the original DataFrame that are not in the modified one based on 'image_link'
missing_rows = original_df[~original_df.index.isin(modified_df.index)]

# Extract the image links of the missing rows
missing_image_links = missing_rows['image_link'].tolist()

# Specify the download folder for the missing images
download_folder = 'downs'
# print(missing_image_links)

# Call the download_images function to download only the missing images
download_images(missing_image_links, download_folder, allow_multiprocessing=False)

print(f"Missing images downloaded: {len(missing_image_links)}")
