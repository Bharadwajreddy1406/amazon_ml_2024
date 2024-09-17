import pandas as pd
from src.utils import download_images

if __name__ == '__main__':
    # Read the CSV file
    # train_df = pd.read_csv(r"dataset/train.csv")
    #
    #
    # # Extract the image links
    # image_links = train_df['image_link'].tolist()
    # image_links = image_links[:30]
    #
    #
    # # Specify the download folder
    # download_folder = 'downloads'
    #
    # # Call the download_images function
    # download_images(image_links, download_folder, allow_multiprocessing=False)

    #
    # Read the CSV file

    import pandas as pd

    # Load the existing CSV file into a DataFrame
    df = pd.read_csv('dataset/train.csv')
    df = df[:25]
    print(df)
    # Check if the DataFrame already has a unique index
    if not df.index.is_unique:
        # Create a new unique index
        df.reset_index(drop=True, inplace=True)
        df.index.name = 'unique_index'

    # Save the updated DataFrame back to the CSV file
    df.to_csv('final_custos/train.csv', index=True)