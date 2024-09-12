import pandas as pd
from src.utils import download_images

if __name__ == '__main__':
    # Read the CSV file
    train_df = pd.read_csv(r"dataset/train.csv")

    # Extract the image links
    image_links = train_df['image_link'].tolist()
    image_links = image_links[:10]

    # Specify the download folder
    download_folder = 'downloads'

    # Call the download_images function
    download_images(image_links, download_folder, allow_multiprocessing=False)