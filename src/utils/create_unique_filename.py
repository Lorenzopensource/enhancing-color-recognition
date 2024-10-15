import os

def create_unique_filename(base_name, extension):
    """
    Create a unique filename by appending a number if the file already exists.
    
    Args:
        base_name (str): The base name for the file.
        extension (str): The file extension (including the dot).

    Returns:
        str: A unique filename.
    """
    count = 1
    unique_name = f"{base_name}{extension}"
    while os.path.exists(unique_name):
        unique_name = f"{base_name}_{count}{extension}"
        count += 1
    return unique_name