import numpy as np
import cv2
from color_limits import get_limits  # Importing get_limits from color_limits.py

def recolor(img, mask, original_color, target_color):
    modified_image = img.copy()
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)

    lower_limit, upper_limit = get_limits(original_color)
    original_range = {'lower': lower_limit, 'upper': upper_limit}

    lower_limit, upper_limit = get_limits(target_color)
    new_range = {'lower': lower_limit, 'upper': upper_limit}

    for i in [0, 1, 2]:
        channel = modified_image[:, :, i]
        current_values = channel.copy()

        original_range_width = original_range['upper'][i] - original_range['lower'][i]
        new_range_width = new_range['upper'][i] - new_range['lower'][i]

        if original_range_width == 0 or new_range_width == 0:
            continue

        new_values = np.clip((current_values - original_range['lower'][i]) * 
                             (new_range_width / original_range_width) + 
                             new_range['lower'][i], 0, 255)

        channel[mask == 1] = new_values[mask > 0].astype(channel.dtype)
        modified_image[:, :, i] = channel

    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_HSV2RGB)
    return modified_image