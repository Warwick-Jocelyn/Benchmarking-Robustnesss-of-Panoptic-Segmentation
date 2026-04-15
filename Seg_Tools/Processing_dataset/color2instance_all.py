import cv2
import numpy as np
import os
import glob

# Set the parent folder path
parent_folder_path = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Backup/Oneformer_for_ACDC_eval/OneFormer/datasets/cityscapes/gtFine/train'

# Define color mapping relationships, converting colors from BGR format to RGB
color_mapping = {
    (60, 20, 220): (93, 93, 93),  # person
    (142, 0, 0): (101, 101, 101),  # car
    (70, 0, 0): (105, 105, 105),  # truck
    (0, 0, 255): (97, 97, 97),  # rider
    (100, 60, 0): (109, 109, 109),  # bus
    (100, 80, 0): (121, 121, 121),  # train
    (230, 0, 0): (125, 125, 125),  # motorcycle
    (32, 11, 119): (128, 128, 128)  # bicycle
}

def map_color_to_instance(img):
    # Create an empty array with the same size as the input image
    height, width, _ = img.shape
    instance_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply color mapping to each pixel
    for key, value in color_mapping.items():
        # Find all pixels that match the key color
        mask = (img == key).all(axis=2)
        # Set the corresponding pixels in the output image to the mapped color
        instance_img[mask] = value

    return instance_img

# Traverse all sub-folders in the parent folder
for sub_folder in os.listdir(parent_folder_path):
    sub_folder_path = os.path.join(parent_folder_path, sub_folder)
    
    if os.path.isdir(sub_folder_path):
        # Iterate over all images matching the criteria in the sub-folder
        for file_path in glob.glob(os.path.join(sub_folder_path, '*_gtFine_labelColor.png')):
            # Read the image
            img = cv2.imread(file_path)

            # Convert the image to instance image
            instance_img = map_color_to_instance(img)

            # Define the new file name
            new_file_name = file_path.replace('_gtFine_labelColor.png', '_gtFine_instanceIds.png')

            # Save the image
            cv2.imwrite(new_file_name, instance_img)

            print(f'Processed and saved: {new_file_name}')

# Output confirmation that all images have been processed
print("All images have been processed.")
