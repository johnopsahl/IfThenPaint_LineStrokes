import os
import math
import json
import numpy as np
import cv2
from definitions import DATA_PATH

def resize_image_for_line_stroke(image: np.ndarray, 
                                 painting_width: float, 
                                 smallest_brush_width: float) -> np.ndarray:
    '''
    Resizes the image resolution to optimize coverage of the line stroke 
    algorithm. The image is resized so that the width of a pixel is 
    representative of the width of the smallest brush.
    
    Args:
        image (CV_8U): input bitmap image
        painting_width: width of the painting in mm
        smallest_brush_width: width of the smallest brush in mm
    
    Returns:
        image_sized (CV_8U): resized bitmap image
    '''
    
    height, width = image.shape[:2]
    aspect_ratio = width/height
    resize_width = math.floor(painting_width/smallest_brush_width)
    resize_height = math.floor(resize_width/aspect_ratio)
    
    image = cv2.resize(image,
                       (resize_width,resize_height), 
                       interpolation=cv2.INTER_AREA)
    
    return image

def color_quantize_image(image: np.ndarray, 
                         color_count: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Reduces image to a fixed number of colors.
    
    Args:
        image
        color_count: number of colors in output image
        
    Returns:
        image_quant: output color quantisized image
        centers: k-means color centers
    '''
    
    if len(image.shape) == 2:
        # Image is grayscale
        z = image.reshape(-1)
    else:
        z = image.reshape((-1, 3))
        
    z = np.float32(z)
    
    # Define criteria, number of clusters(k) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, label, centers = cv2.kmeans(z, 
                                             color_count, 
                                             None, 
                                             criteria, 
                                             10, 
                                             cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to unit8 and make original image
    centers = np.uint8(centers)
    image_colorquant = centers[label.flatten()]
    image_colorquant = image_colorquant.reshape((image.shape))
    image = image_colorquant
    
    return image, centers

def convert_bgr_pixel_color_to_grayscale(pixel_bgr: list[float]) -> float:
    '''
    Converts single bgr pixel to grayscale using the weighted luminosity method
    
    Args:
        pixel_bgr (int): [b, g, r] pixel values
        
    Returns:
        pixel_grayscale (int): [grayscale] pixel value
    '''
    
    pixel_grayscale = (0.114*pixel_bgr[0] + 0.587*pixel_bgr[1] 
                       + 0.299*pixel_bgr[2])
    
    return pixel_grayscale

def sort_color_centers_dark_to_light(color_centers: np.ndarray) -> np.ndarray:
    '''
    Sorts color centers from dark to light (via grayscale coversion) 
    so dark colors are painted first.
    
    Args:
        color_centers (int): [b, g, r] color quantized image color centers
    
    Returns:
        color_centers_sorted (int): [b, g, r] color quantized image color
            centers ordered from dark to light
    '''
    
    if color_centers.shape[1] == 1:
        # Image is grayscale
        color_sorted = sorted(color_centers)
    
    else:
        # Image is not grayscale
        grayscale_centers = np.uint8([convert_bgr_pixel_color_to_grayscale(x) 
                                      for x in color_centers])
    
        combined = zip(grayscale_centers, color_centers)
        sorted_combined = sorted(combined)
    
        grayscale_sorted, color_sorted = zip(*sorted_combined)
          
    color_centers_sorted = np.uint8(color_sorted)
    
    return color_centers_sorted
    
if __name__ == '__main__':
    
    # Import image
    image = cv2.imread(os.path.join(DATA_PATH, 'Peppers.png'))
    
    # Convert image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_resize = resize_image_for_line_stroke(image, 
                                                8*25.4, 
                                                0.125*25.4)
    
    cv2.imwrite(os.path.join(DATA_PATH, 'image_resize.png'), image_resize)
    
    color_count = 8
    
    # Use color_count + 1 for when removing a white background
    image_quant, color_centers_quant = color_quantize_image(image_resize, 
                                                            color_count + 1)
    
    color_centers = sort_color_centers_dark_to_light(
                            color_centers_quant)
    
    # Remove the white background color so it does not get painted
    # color_centers_list = color_centers_sorted.tolist()
    # color_centers = np.uint8(color_centers_list[:-1])
    
    cv2.imwrite(os.path.join(DATA_PATH, 'image_quant.png'), image_quant)
    
    # Create an image of each color center
    for i in range(len(color_centers)):
        color_mask = cv2.inRange(image_quant, 
                                 color_centers[i], 
                                 color_centers[i])
        image_color = cv2.bitwise_and(image_quant, image_quant, 
                                      mask = color_mask)
        cv2.imwrite(os.path.join(DATA_PATH, 'color_center_' + str(i) + '.png'), 
                    image_color)
        
    # Create a binary image of each color center
    for i in range(len(color_centers)):
        image = cv2.imread(os.path.join(DATA_PATH, 'color_center_' + str(i) 
                                        + '.png'))
        ret, binary_image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(DATA_PATH, 'color_center_' + str(i) + 
                               '_binary.png'), binary_image)
    
    # Write color_center list to json file
    with open(os.path.join(DATA_PATH, 'image_color_center_bgr.txt'),'w') as f:
        json.dump(color_centers.tolist(), f, separators = (',', ':'), 
                  sort_keys = True, indent = 4)
    f.close()
    
    