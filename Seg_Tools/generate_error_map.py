import torch
import numpy as np
from PIL import Image

# Load the NumPy array from the CSV file

numpy_array = np.loadtxt('/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/OneFormer_ACDC_eval_with_TTA/pre_class_map.csv', delimiter=',', dtype=int)

numpy_array2 = np.loadtxt('/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/OneFormer_ACDC_eval_with_TTA/Jocelyn/output.csv', delimiter=',', dtype=int)

# Convert the NumPy array back to a tensor
cur_mask_ids = torch.tensor(numpy_array)

gt_mask_ids = torch.tensor(numpy_array2)


# If you need the tensor back on the CUDA device
cur_mask_ids = cur_mask_ids.to('cuda').cpu()
gt_mask_ids = gt_mask_ids.to('cuda').cpu()


difference_tensor = np.where(cur_mask_ids != gt_mask_ids, 1, 0)

binary_mask = (difference_tensor * 255).astype(np.uint8)

image = Image.fromarray(binary_mask)

image.save('error_map.png')

print(difference_tensor)
