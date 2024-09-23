from PIL import Image
import numpy as np
from recoloring_functions import recolor

def main():
    # Load the image
    img = np.array(Image.open('examples/original.png')) #Load your image in RGB format

    # Define your binary mask
    mask = np.ones(img.shape[:2], dtype=np.uint8)  # Example: full mask

    # Define the original and the target color
    original_color = 'blue'
    target_color = 'red' 


    result = recolor(img, mask, original_color, target_color) #The output image is in RGB format

    # Save the result
    result_image = Image.fromarray(result)
    result_image.save(f'recolored_to_{target_color}.jpg')
    print(f"Image saved as 'recolored_to_{target_color}.jpg'")

if __name__ == "__main__":
    main()