import numpy as np

# Define color limits in HSV colorspace
color_limits = {
    'red': [np.array([0, 70, 70]), np.array([4.4, 255, 255])],
    'red2': [np.array([175, 110, 140]), np.array([179, 255, 255])],
    'green': [np.array([47, 70, 117.3]), np.array([65, 255, 255])],
    'blue': [np.array([108, 70, 70]), np.array([117, 255, 255])],
    'cyan': [np.array([88.5, 70, 76]), np.array([95, 255, 255])],
    'orange': [np.array([12, 70, 100]), np.array([19, 255, 255])],
    'purple': [np.array([137, 70, 84]), np.array([142, 255, 255])],
    'pink': [np.array([157, 70, 100]), np.array([167, 200, 255])],
    'brown': [np.array([15, 40, 30]), np.array([19, 255, 190])],
    'yellow': [np.array([28.5, 70, 100]), np.array([32, 255, 255])],
    'black': [np.array([0, 0, 0]), np.array([179, 3, 130])],
    'white': [np.array([0, 0, 195]), np.array([179, 3, 255])],
    'gray': [np.array([0, 0, 40]), np.array([179, 3, 190])]
}

def get_limits(color):
    lower_limit = color_limits[color][0]
    upper_limit = color_limits[color][1]
    return lower_limit, upper_limit