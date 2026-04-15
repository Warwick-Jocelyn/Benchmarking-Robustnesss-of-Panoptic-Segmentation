import os
from PIL import Image

def convert_24bit_to_32bit(image_path, output_path):
    # Load the 24-bit image
    image = Image.open(image_path).convert('RGB')  # Ensure it's in 24-bit RGB mode

    # Get the size of the image
    width, height = image.size

    # Create a new image with an alpha channel (RGBA)
    image_32bit = Image.new('RGBA', (width, height))

    # Copy the RGB data and add the alpha channel
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            image_32bit.putpixel((x, y), (r, g, b, 255))

    # Save the new image
    image_32bit.save(output_path)
    return output_path

def process_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            convert_24bit_to_32bit(input_image_path, output_image_path)
            print(f"Processed {input_image_path} -> {output_image_path}")

# Example usage
input_folder = "/home/haonan_zhao/jocelyn/OneFormer/datasets/cityscapes/gtFine/val/GP010476"
output_folder = "/home/haonan_zhao/jocelyn/OneFormer/datasets/cityscapes/gtFine/val/GOPR0476"
process_folder(input_folder, output_folder)
