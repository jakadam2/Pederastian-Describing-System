from PIL import Image
import os
import argparse

def convert_png_to_jpg():
    # argparser 
    parser = argparse.ArgumentParser(description='input and output folders')
    parser.add_argument('--png', help='path of the input png folder')
    parser.add_argument('--jpg', help='path of the output jpg folder')
    args = parser.parse_args()
    png_folder = args.png
    jpg_folder = args.jpg

    # Create the output folder if it doesn't exist
    if jpg_folder == None:
        jpg_folder = png_folder+'_jpg'
        print(jpg_folder)
        os.makedirs(jpg_folder)

    # Loop through each PNG file in the input folder
    for file_name in os.listdir(png_folder):
        if file_name.endswith(".png"):
            # Open the PNG file
            png_path = os.path.join(png_folder, file_name)
            img = Image.open(png_path)

            # Create the output JPG file path
            jpg_path = os.path.join(jpg_folder, os.path.splitext(file_name)[0] + ".jpg")

            # Convert and save as JPG
            img.convert("RGB").save(jpg_path)

def main(): 
    convert_png_to_jpg()

if __name__ == "__main__":
    # Call the function to convert PNG to JPG
    main()


