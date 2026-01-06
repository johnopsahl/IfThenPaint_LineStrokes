import os
import math
import random
import json
import numpy as np
import cv2
import svgwrite
from definitions import DATA_PATH


def calculate_scan_origin_pixel_index(image_pixel_height: int,
                                      image_pixel_width: int) -> list[int]:
    '''
    Determine the scan origin pixel position. The scan origin pixel is the 
    center of the line scan.
    
    Args:
        image pixel height
        image pixel width
    
    Returns:
        scan_origin_pixel_index: (y, x) pixel position 
            of the scan origin pixel
    '''

    #minus one because image pixel positions are zero indexed
    scan_origin_pixel_index = [round((image_pixel_height/2) - 1),
                               round((image_pixel_width/2) - 1)]
    
    return scan_origin_pixel_index

def convert_pixel_index_to_coord(pixel_index: list[int]) -> list[float]:
    '''
    Convert a pixel index to a pixel coordinate.
    
    Args:
        pixel_index: (y, x) pixel index
    
    Returns:
        pixel_coord: (x, y) pixel coordinates
    
    '''
    
    #image pixel index is [0, 0] at top left corner of image
    pixel_coord = [pixel_index[1] + 0.5, pixel_index[0] + 0.5]
    
    return pixel_coord

def create_pixel_coords(image_pixel_height: int, 
                        image_pixel_width: int) -> list[list[float]]:
    '''
    Create list of same height and width dimensions as image, 
    with 3rd dimension as pixel coordinates.
    
    Args:
        image_pixel_height
        image_pixel_width
    
    Returns:
        pixel_coords
    '''
    
    pixel_coords = []
    for i in range(image_pixel_height):
        row = []
        for j in range(image_pixel_width):
            row.append(convert_pixel_index_to_coord([i,j]))
        pixel_coords.append(row)
        
    return pixel_coords

def translate_coord_to_origin(input_coord: list[float], 
                              new_origin_coord: list[float]) -> list[float]:
    '''
    Translates coordinates by the origin coordinates.
    
    Args:
        input_coord: (x, y) input coordinates
        new_origin_coord: (x, y) origin coordinates
        
    Returns:
        translated_coord: (x, y) coordinates translated 
            about the origin coordinates
    '''
    
    translated_coord = [input_coord[0] - new_origin_coord[0],
                        input_coord[1] - new_origin_coord[1]]
    
    return translated_coord

def rotate_coord_about_origin(input_coord: list[float], 
                              rotation_angle: float) -> list[float]:
    '''
    Rotates coordinates about the origin coordinates.
    
    Args:
        input_coord: (x, y) input coordinates
        rotation_angle: CCW angle of rotation about the origin coordinates
        
    Returns:
        rotated_coord: (x, y) coordinates rotated about the origin (0, 0)
    '''
    
    theta = math.radians(rotation_angle)
    x_rot = (input_coord[0]*math.cos(theta) - input_coord[1]*math.sin(theta))
    y_rot = (input_coord[0]*math.sin(theta) + input_coord[1]*math.cos(theta))
    rotated_coord = [x_rot, y_rot]
    
    return rotated_coord

def calc_nested_scan_offsets(scan_width: float) -> list[float]:
    '''
    Calculate a list of nested scan line offsets for 
    a given scan line width.
    
    Args:
        scan_width
    
    Returns:
        scan_offsets: a list of all scan offsets that are to the 
            evaluated for nested placement of line scans
    '''
    
    scan_offsets = []
    
    scan_offset = -scan_width/2 + 0.5
    
    while scan_offset < scan_width/2:
        scan_offsets.append(scan_offset)
        scan_offset += 1
        
    return scan_offsets

def calc_scan_y_coords(scan_width: float, scan_offset: float, 
                       rotated_pixel_coords: list[list[float]]) -> list[float]:
    '''
    Calculate a list of scan line y coordinates.
    
    Args:
        scan_width
        scan_offset
        rotated_image_coords
    
    Returns:
        scan_line_y_coords: list of scan line y coordinates
    '''
    
    y_coords = [coord[1] for row in rotated_pixel_coords for coord in row]
    
    # 0.5 adjustment because min pixel coord is 0.5 but coordinate space 
    # is valid down to 0
    y_coord_min = min(y_coords) - 0.5
    y_coord_max = max(y_coords) + 0.5
    
    y_pos_initial = (scan_width/2) - scan_offset # (1/2) - (-) = 2
    y_neg_initial = -(scan_width/2) - scan_offset # -(2/2) - (-1) = 0
    y_center_initial = (y_pos_initial + y_neg_initial)/2
    
    num_pos_scans = math.floor((y_coord_max - y_pos_initial)/scan_width)
    num_neg_scans = math.floor((y_neg_initial - y_coord_min)/scan_width)
    
    y_center_top = y_center_initial + scan_width*num_pos_scans
    y_center_btm = y_center_initial - scan_width*num_neg_scans
    
    scan_y_coords = []
    scan_y = y_center_btm
    
    while scan_y <= y_center_top:
        scan_y_coords.append(scan_y)
        scan_y += scan_width
    
    return scan_y_coords

def identify_scan_pixels(
        scan_width: float, 
        scan_y_coords: list[float], 
        rotated_pixel_coords: list[list[float]]) -> tuple[list[list[int]], 
                                                          list[list[float]]]:
    '''
    Determine pixels of the rotated image that are within each scan line area.
    
    Args:
        scan_width
        scan_y_coords
        rotated_pixel_coords
            
    Returns:
        scan_pixel_indices: list of indices for pixels contained within
            the scan line
        scan_pixel_coords: list of coordinates for pixels contained
            within the scan line
    '''
    scan_pixel_indices = []
    scan_pixel_coords = []
    
    height = len(rotated_pixel_coords)
    width = len(rotated_pixel_coords[0])
    
    for i in range(len(scan_y_coords)):
        y_top = scan_y_coords[i] + (scan_width/2)
        y_btm = scan_y_coords[i] - (scan_width/2)
    
        temp_scan_pixel_indices = []
        temp_scan_pixel_coords = []
        for j in range(height):
            for k in range(width):
                if y_btm <= rotated_pixel_coords[j][k][1] < y_top:
                    temp_scan_pixel_indices.append([j, k])
                    temp_scan_pixel_coords.append(
                        [rotated_pixel_coords[j][k][0],
                        rotated_pixel_coords[j][k][1]])
        
        scan_pixel_indices.append(temp_scan_pixel_indices)
        scan_pixel_coords.append(temp_scan_pixel_coords)
    
    return scan_pixel_indices, scan_pixel_coords

def eval_scan_x_range(
        scan_pixel_coords: list[list[float]]) -> tuple[list[float], 
                                                       list[float]]:
    '''
    Evaluate minimum and maximum x coordinate of each scan line pixel set. 
    
    Args:
        scan_pixel_coords: [x, y] pixel coordinates for all pixels in
            a single line scan
    
    Returns:
        x_min: minimum x coordinate value of all pixels in the scan 
            line pixel set
        x_max: maximum x coordinate value of all pixels in the scan
            line pixel set
    '''
    
    x_min = []
    x_max = []
    for i in range(len(scan_pixel_coords)):
        x_temp = []
        for j in range(len(scan_pixel_coords[i])):
                x_temp.append(scan_pixel_coords[i][j][0])
        x_min.append(min(x_temp))
        x_max.append(max(x_temp))
        
    return x_min, x_max

def write_lines_to_svg(filename: str, view_box: list[float], 
                       line_end_points: list[list[float]], scan_width: float):
    '''
    Write scan lines to svg.
    
    Args:
        filename: output svg file name
        view_box: [minx, miny, width, height] output svg 
            view box parameters
        line_end_points
        scan_width
    
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    dwg = svgwrite.Drawing(os.path.join(DATA_PATH, filename + '.svg'))
    for i in range(len(line_end_points)):
        x1 = line_end_points[i][0][0]
        y1 = line_end_points[i][0][1]
        x2 = line_end_points[i][1][0]
        y2 = line_end_points[i][1][1]
        dwg.add(dwg.line((x1, y1), (x2, y2), 
                         stroke="black", 
                         stroke_width=scan_width - 0.02))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()
    
def write_scnwdt_lines_to_svg(filename: str, view_box: list[float], 
                              line_strokes: list[list[float]], 
                              scan_widths: list[float]):
    '''
    Write scan lines to svg with lines of multiple scan widths.
    
    Args:
        filename: output svg file name
        view_box: [minx, miny, width, height] output svg 
            view box parameters
        line_strokes
        scan_widths
        
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    dwg = svgwrite.Drawing(os.path.join(DATA_PATH, filename + '.svg'))
    for i in range(len(line_strokes)):
        
        scan_width = scan_widths[i]
        r_rgb = random.randint(1,255)
        g_rgb = random.randint(1,255)
        b_rgb = random.randint(1,255)
        
        for j in range(len(line_strokes[i])):
            x1 = line_strokes[i][j][0][0]
            y1 = line_strokes[i][j][0][1]
            x2 = line_strokes[i][j][1][0]
            y2 = line_strokes[i][j][1][1]
            dwg.add(dwg.line((x1, y1), (x2, y2), 
                             stroke=svgwrite.rgb(r_rgb, g_rgb, b_rgb), 
                             stroke_width=scan_width - 0.02))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()
    
def write_color_centers_line_strokes_to_svg(
        view_box: list[float], image_color_centers: list[list[int]],
        line_end_points: list[list[float]], scan_width: float):
    '''
    Write color center line strokes to svg.
    
    Args:
        view_box: [minx, miny, width, height] output svg 
            view box parameters
        image_color_centers
        line_end_points
        scan_width
    
    Returns:
       Outputs a svg image to the data folder. 
    '''
    
    dwg = svgwrite.Drawing(os.path.join(DATA_PATH, 
                                        'color_center_line_strokes.svg'))
    for i in range(len(image_color_centers)):
        
        # Image color centers are formatted brg so need to unpack in reverse
        r_rgb = image_color_centers[i][2]
        g_rgb = image_color_centers[i][1]
        b_rgb = image_color_centers[i][0]
        
        for j in range(len(line_end_points[i])):
            x1 = line_end_points[i][j][0][0]
            y1 = line_end_points[i][j][0][1]
            x2 = line_end_points[i][j][1][0]
            y2 = line_end_points[i][j][1][1]
            dwg.add(dwg.line((x1, y1), (x2, y2), 
                             stroke=svgwrite.rgb(r_rgb, g_rgb, b_rgb), 
                             stroke_width=scan_width - 0.02))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

def write_color_centers_scnwdt_strokes_to_svg(
        filename: str, view_box: list[float], 
        image_color_centers: list[list[int]],
        color_centers_strokes_list: list[list[list[float]]], 
        color_centers_scan_width_list: list[list[float]]):
    '''
    Write color center scan width line strokes to svg.
    
    Args:
        filename: output svg file name
        view_box: [minx, miny, width, height] output svg 
            view box parameters
        image_color_centers
        color_centers_strokes_list
        color_centers_scan_width_list
    
    Returns:
       Outputs a svg image to the data folder.  
    '''
    
    dwg = svgwrite.Drawing(os.path.join(DATA_PATH, filename + '.svg'))
    
    for i in range(len(color_centers_strokes_list)):
        # For each color center
        
        if len(image_color_centers[0]) == 1:
            # Image is grayscale
            r_rgb = image_color_centers[i][0]
            g_rgb = image_color_centers[i][0]
            b_rgb = image_color_centers[i][0]
        else: 
            # Image is not grayscale
            # Opencv image color centers are formatted brg so need to
            # unpack in reverse
            r_rgb = image_color_centers[i][2]
            g_rgb = image_color_centers[i][1]
            b_rgb = image_color_centers[i][0]
        
        for j in range(len(color_centers_strokes_list[i])):
            #for each scan width
            
            scan_width = color_centers_scan_width_list[i][j]
            
            for k in range(len(color_centers_strokes_list[i][j])):
                #for each line stroke
                
                x1 = color_centers_strokes_list[i][j][k][0][0]
                y1 = color_centers_strokes_list[i][j][k][0][1]
                x2 = color_centers_strokes_list[i][j][k][1][0]
                y2 = color_centers_strokes_list[i][j][k][1][1]
                dwg.add(dwg.line((x1, y1), (x2, y2), 
                                 stroke=svgwrite.rgb(r_rgb, g_rgb, b_rgb), 
                                 stroke_width=scan_width - 0.02))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()
    
def sort_indices_and_coords_by_x_coord(
        scan_pixel_indices: list[list[int]], 
        scan_pixel_coords: list[list[float]]) -> tuple[list[list[int]],
                                                       list[list[float]]]:
    '''
    Sort scan pixel indices and scan pixel coords by increasing
    x coordinate value of scan pixel coords. Establishes line scan 
    sequence from negative to positive x coordinate.
    
    Args:
        scan_pixel_indices (int)
        scan_pixel_coords (float)
    
    Returns:
        sorted_scan_pixel_indices (int)
        sorted_scan_pixel_coords (float)
    '''
    
    sorted_scan_pixel_indices = []
    sorted_scan_pixel_coords = []
    for i in range(len(scan_pixel_coords)):
        x_coords = [x for x, y in scan_pixel_coords[i]]
        indexed_x_coords = list(enumerate(x_coords))
        sorted_x_coords = sorted(indexed_x_coords, key=lambda item: item[1])
        sorted_x_indices = [index for index, value in sorted_x_coords]
        sorted_indices = [scan_pixel_indices[i][j] for j in sorted_x_indices]
        sorted_coords = [scan_pixel_coords[i][j] for j in sorted_x_indices]
        sorted_scan_pixel_indices.append(sorted_indices)
        sorted_scan_pixel_coords.append(sorted_coords)
    
    return sorted_scan_pixel_indices, sorted_scan_pixel_coords

def determine_scan_pixel_seq_groups(
        sorted_scan_pixel_coords: list[list[float]]) -> list[list[int]]:
    '''
    Create sequence groups for all pixels contained a scan line.
    If multiple pixels (that are contained within the same scan line) share 
    the same x coordinates, assign those pixels to the same sequence group. 
    Sequence groups contain the index data of scan line pixels and not the 
    pixel coordinate data. 
    
    Args:
        sorted_scan_pixel_coords
        
    Returns:
        scan_pixel_seq_groups
    '''
    
    scan_pixel_seq_groups = []
    for i in range(len(sorted_scan_pixel_coords)): 
        # For each scan line
        
        line_seq = []
        seq_group_start = None
        seq_group_end = None
        for j in range(len(sorted_scan_pixel_coords[i])): 
            # For each pixel
            
            if j == 0: 
                #first pixel in list
                x_prev = None
            else:
                x_prev = sorted_scan_pixel_coords[i][j - 1][0]
                
            x_current = sorted_scan_pixel_coords[i][j][0]
            
            if j + 1 > len(sorted_scan_pixel_coords[i]) - 1:
                #last pixel in list
                x_next = None
            else:
                x_next = sorted_scan_pixel_coords[i][j + 1][0]
                
            if x_current != x_prev:
                seq_group_start = j
                
            if x_current != x_next: 
                seq_group_end = j
                line_seq.append([seq_group_start, seq_group_end])
                
        scan_pixel_seq_groups.append(line_seq)
                
    return scan_pixel_seq_groups

def validate_scan_pixel_seq_groups(
        binary_image: np.ndarray,
        scan_pixel_indices: list[list[int]],
        scan_pixel_seq_groups: list[list[int]]) -> list[list[bool]]:
    '''
    Evaluate scan pixel sequence groups against the binary image. Scan pixel 
    groups that match the binary image are valid. 
    
    Args:
        binary_image
        scan_pixel_indices
        scan_pixel_seq_groups
        
    Returns:
        scan_pixel_seq_groups_valid
    '''
    
    scan_pixel_seq_groups_valid = []
    for i in range(len(scan_pixel_seq_groups)): 
        # For each scan line
        line_seq_groups_valid = []
        for j in range(len(scan_pixel_seq_groups[i])): 
            # For each pixel sequence group
            
            start_pixel_index = scan_pixel_seq_groups[i][j][0]
            end_pixel_index = scan_pixel_seq_groups[i][j][1]
            
            # If any one pixel of the sequence group is not valid against the
            # binary image, then all pixels in the sequence group are not valid
            seq_group_valid = True
            for k in range(start_pixel_index, end_pixel_index + 1):
                # For each pixel of the pixel sequence group
                y_index = scan_pixel_indices[i][k][0]
                x_index = scan_pixel_indices[i][k][1]
                
                if binary_image[y_index][x_index][0] != 255:
                    seq_group_valid = False
            line_seq_groups_valid.append(seq_group_valid)
        scan_pixel_seq_groups_valid.append(line_seq_groups_valid)
    
    return scan_pixel_seq_groups_valid

def identify_line_strokes(
        binary_image: np.ndarray,
        scan_y_coords: list[float],
        scan_pixel_indices: list[list[int]],
        scan_pixel_coords: list[list[float]],
        scan_pixel_seq_groups: list[list[int]],
        scan_pixel_seq_groups_valid: list[list[float]]) -> tuple[list[list[float]],
                                                                 list[list[int]]]:
    '''
    Identify line strokes using the the binary image and 
    scan line pixel data.
    
    Args:
        binary_image
        scan_y_coords
        scan_pixel_indices
        scan_pixel_coords
        scan_pixel_seq_groups
        scan_pixel_seq_groups_valid
        
    Returns:
        line_strokes: [[x_start, y_start],[x_end, y_end]] end point
            coordinates for each line stroke
        line_stroke_pixel_indices: all pixels contained within each
            line stroke
    '''
    
    line_strokes = []
    line_strokes_pixel_indices = []
    
    for i in range(len(scan_pixel_seq_groups)): 
        # For each scan line
        
        line_started = False
        temp_pixel_indices = []
        
        for j in range(len(scan_pixel_seq_groups[i])):
            # For each pixel sequence group
            
            group_start_index = scan_pixel_seq_groups[i][j][0]
            group_end_index = scan_pixel_seq_groups[i][j][1]
            
            if scan_pixel_seq_groups_valid[i][j] == True:
                # If pixel sequence group is a valid match to binary image
                
                # Line stroke pixel logic
                for k in range(group_start_index, group_end_index + 1):
                    temp_pixel_indices.append(scan_pixel_indices[i][k])
                    
                # Line stroke logic
                if line_started == False:
                    #if line has not been started, start the line
                    start_x_coord = scan_pixel_coords[i][group_start_index][0]
                    line_started = True
                
                if line_started == True:
                    # If line has been started, continue the line
                    end_x_coord = scan_pixel_coords[i][group_start_index][0]
                
                    if j == len(scan_pixel_seq_groups[i]) - 1:
                        #If the last pixel sequence group in the scan, 
                        # end the line
                        start_point = [start_x_coord, scan_y_coords[i]]
                        end_point = [end_x_coord, scan_y_coords[i]]
                        line_strokes.append([start_point, end_point])
                        line_started = False
                        line_strokes_pixel_indices.append(temp_pixel_indices)
                        temp_pixel_indices = []
                        
            else:
                # If invalid pixel sequence group
                
                if line_started == True:
                    # If line has been started, end the line
                    start_point = [start_x_coord, scan_y_coords[i]]
                    end_point = [end_x_coord, scan_y_coords[i]]
                    line_strokes.append([start_point, end_point])
                    line_started = False
                    line_strokes_pixel_indices.append(temp_pixel_indices)
                    temp_pixel_indices = []
                
    return line_strokes, line_strokes_pixel_indices

def apply_line_stroke_end_condition(
        line_strokes: list[list[float]]) -> list[list[float]]:
    '''
    Apply 0.5 extension to both ends of each line stroke. 
    Beacuse line strokes are evaluated according to pixel centers, 
    need to add end condition to ensure full coverage of the pixels 
    at the beginning and end of each lines.
    
    Args:
        line_strokes: [[x_start, y_start],[x_end, y_end]] end point
            coordinates for each line stroke
        
    Returns: 
        line_strokes_end_condition: end condition applied to
            line strokes
    '''
    
    line_end_extension = 0.5
    
    line_strokes_end_condition = line_strokes.copy()
    for i in range(len(line_strokes_end_condition)):
        line_strokes_end_condition[i][0][0] -=  line_end_extension
        line_strokes_end_condition[i][1][0] += line_end_extension
        
    return line_strokes_end_condition

def long_pass_line_strokes(
        line_strokes: list[list[float]], 
        line_strokes_pixel_indices: list[list[int]],
        min_stroke_length: float) -> tuple[list[list[float]],
                                           list[list[int]]]:
    '''
    Return line strokes (and associated line strokes pixel indices)
    that are longer than the specified minimum length.
    
    Args:
        line_strokes (float)
        line_strokes_pixel_indices (int)
        min_stroke_length (float)
    
    Returns:
        line_strokes_long (float)
        line_strokes_pixel_indices_long (int)
    '''
    
    line_strokes_long = []
    line_strokes_pixel_indices_long = []
    
    for i in range(len(line_strokes)):
        start_point = line_strokes[i][0]
        end_point = line_strokes[i][1]
        line_length = calc_line_length(start_point, end_point)
        if (line_length >= min_stroke_length -0.1 or 
            math.isclose(line_length, min_stroke_length)):
            #math.isclose() corrects for floating point error
            
            line_strokes_long.append(line_strokes[i])
            line_strokes_pixel_indices_long.append(
                line_strokes_pixel_indices[i])
    
    return line_strokes_long, line_strokes_pixel_indices_long

def calc_line_length(start_point: list[float], 
                     end_point: list[float]) -> float:
    '''
    Calculate Euclidean distance between two points.
    
    Args:
        start_point: [x, y] start point coordinates
        end_point: [x, y] end point coordinates
    
    Returns:
        line_length
    '''
    
    line_length = ((end_point[0] - start_point[0])**2 + 
                   (end_point[1] - start_point[1])**2)**0.5
    
    return line_length

def scan_line_strokes(binary_image: np.ndarray, 
                      scan_width: float,
                      scan_offset: float, 
                      scan_angle: float) -> tuple[list[list[float]],
                                                  list[list[int]]]: 
    '''
    Scan a binary image for line strokes.
    
    Args:
        binary_image
        scan_width
        scan_offset
        scan_angle
    
    Returns:
        line_strokes: [[x_start, y_start],[x_end, y_end]] end point
            coordinates for each line stroke
        line_stroke_pixel_indices: indices of pixels contained within
            each line stroke
    '''
    
    image_height, image_width = binary_image.shape[:2]
    
    scan_origin_index = calculate_scan_origin_pixel_index(image_height, 
                                                          image_width)
    
    scan_origin_coord = convert_pixel_index_to_coord(scan_origin_index)
    
    pixel_coords = create_pixel_coords(image_height, image_width)
    
    translated_coords = pixel_coords.copy()
    for i in range(image_height):
        for j in range(image_width):
            translated_coords[i][j] = translate_coord_to_origin(
                                        translated_coords[i][j],
                                        scan_origin_coord)
    
    rotated_coords = translated_coords.copy()
    for i in range(image_height):
        for j in range(image_width):
            rotated_coords[i][j] = rotate_coord_about_origin(
                                    rotated_coords[i][j],
                                    scan_angle)
    
    scan_y_coords = calc_scan_y_coords(scan_width, scan_offset, rotated_coords)
    
    (scan_pixel_indices, 
     scan_pixel_coords) = identify_scan_pixels(scan_width,
                                               scan_y_coords, 
                                               rotated_coords)
    
    (sorted_scan_pixel_indices,
     sorted_scan_pixel_coords) = sort_indices_and_coords_by_x_coord(
                                     scan_pixel_indices,
                                     scan_pixel_coords)
    
    scan_pixel_seq_groups = determine_scan_pixel_seq_groups(
                                sorted_scan_pixel_coords)
    
    scan_pixel_seq_groups_valid = validate_scan_pixel_seq_groups(
                                    binary_image,
                                    sorted_scan_pixel_indices,
                                    scan_pixel_seq_groups)
    
    (line_strokes_id, 
     line_strokes_pixel_indices_id) = identify_line_strokes(
                                            binary_image,
                                            scan_y_coords,
                                            sorted_scan_pixel_indices,
                                            sorted_scan_pixel_coords,
                                            scan_pixel_seq_groups,
                                            scan_pixel_seq_groups_valid)
    
    line_strokes_end_condition = apply_line_stroke_end_condition(
                                    line_strokes_id)
    
    # Remove line strokes with length less than the scan_width, to prevent
    # strokes that are wider than thay are long. 
    (line_strokes_long, 
     line_strokes_pixel_indices_long) = long_pass_line_strokes(
                                         line_strokes_end_condition,
                                         line_strokes_pixel_indices_id,
                                         scan_width)
    
    # Unrotate line_strokes
    rotated_line_strokes = []
    for i in range(len(line_strokes_long)):
        start = rotate_coord_about_origin(line_strokes_long[i][0], 
                                          -scan_angle)
        end = rotate_coord_about_origin(line_strokes_long[i][1],
                                        -scan_angle)
        rotated_line_strokes.append([start, end])
    
    # Untranslate scan_line_end_points
    translated_line_strokes = []
    for i in range(len(rotated_line_strokes)):
        start = translate_coord_to_origin(rotated_line_strokes[i][0],
                                          [-scan_origin_coord[0],
                                           -scan_origin_coord[1]])
        end = translate_coord_to_origin(rotated_line_strokes[i][1],
                                        [-scan_origin_coord[0],
                                         -scan_origin_coord[1]])
        translated_line_strokes.append([start, end])
        
    line_strokes = translated_line_strokes.copy()
    line_strokes_pixel_indices = line_strokes_pixel_indices_long.copy()
    
    return line_strokes, line_strokes_pixel_indices

def remove_line_stroke_pixels(
        image: np.ndarray,
        line_strokes_pixel_indices: list[list[int]]) -> np.ndarray:
    '''
    Remove line stroke pixels from an image. 
    
    Args:
        image
        line_stroke_pixel_indices
    
    Returns:
        image_pixels_removed
    '''
    
    image_pixels_removed = image.copy() 
    
    for i in range(len(line_strokes_pixel_indices)): 
        # For each line stroke
        
        for j in range(len(line_strokes_pixel_indices[i])): 
            # For each pixel in the line stroke
            
            y_index = line_strokes_pixel_indices[i][j][0]
            x_index = line_strokes_pixel_indices[i][j][1]
            image_pixels_removed[y_index, x_index] = [0, 0, 0]
    
    # for troubleshooting
    # cv2.imwrite(os.path.join(DATA_PATH, 'removed_line_stroke_pixels.png'),
    #                          image)
    
    return image_pixels_removed

def calc_line_strokes_length(
        line_strokes: list[list[float]]) -> list[float]:
    '''
    Calculate and return a list of line stroke lengths.
    
    Args:
        line_strokes: [[x_start, y_start],[x_end, y_end]] end point
            coordinates for each line stroke
    
    Returns:
        line_strokes_length: length of each line stroke
    '''
    
    line_strokes_length = []
    for i in range(len(line_strokes)):
        start_point = line_strokes[i][0]
        end_point = line_strokes[i][1]
        line_length = calc_line_length(start_point, end_point)
        line_strokes_length.append(line_length)
        
    return line_strokes_length
        
def calc_line_strokes_area(
        line_strokes_length: list[float], scan_width: float) -> float:
    '''
    Calculate and return a list of line stroke areas.
    
    Args:
        line_strokes_length: length of each line stroke
        scan_width
        
    Returns:
        line_strokes_area: area of each line stroke
    ''' 
    
    line_strokes_area = [length * scan_width for length in line_strokes_length]
    
    return line_strokes_area

def calc_line_strokes_total_area(
        line_strokes_area: list[float]) -> float:
    '''
    Calculate the total area covered by all the line strokes. 
    
    Args: 
        line_strokes_area: area of each line stroke
    
    Returns: 
        line_strokes_total_area: sum of all line stroke areas
    '''
    
    line_strokes_total_area = sum(line_strokes_area)
    
    return line_strokes_total_area

def calc_line_strokes_average_area(
        line_strokes_area: list[float]) -> float:
    '''
    Caculate the average area covered by all the line strokes. 
    
    Args:
        line_strokes_area: area of each line stroke
    
    Returns:
        line_strokes_avg_area: average of all line stroke areas
    '''
    
    if not line_strokes_area:
        return 0 #avoids division by zero error for empty lists
    line_strokes_avg_area = sum(line_strokes_area)/len(line_strokes_area)
    
    return line_strokes_avg_area

def select_line_strokes(scan_image: np.ndarray, 
                        scan_widths: list[float], 
                        scan_offsets: list[float], 
                        scan_angles: list[float]) -> tuple[list[list[float]],
                                                           list[float]]:
    '''
    Generate and select the line strokes. 
    
    Args:
        scan_image
        scan_widths
        scan_offsets
        scan_angles
    
    Returns:
        strokes_select_list: list of selected line strokes
        scan_width_select_list: scan width of selected line strokes
    '''
    
    #calculate intial minimum stroke length
    image_height, image_width = scan_image.shape[:2]
    max_stroke_length = (image_height**2 + image_width**2)**0.5
    phi = (1 + 5**0.5)/2 #golden ratio
    min_stroke_length = max_stroke_length/phi
    
    strokes_select_list = []
    scan_width_select_list = []
    min_stroke_length_flag = True

    while min_stroke_length_flag == True:
        
        print(min_stroke_length)
        
        (strokes_scnwdt,
         strokes_pixel_indices_scnwdt) = generate_scnwdt_line_strokes(
                                             scan_image,
                                             scan_widths,
                                             scan_offsets,
                                             scan_angles)
        
        (strokes_long_scnwdt,
         strokes_pixel_indices_long_scnwdt) = long_pass_scnwdt_line_strokes(
                                                 strokes_scnwdt,
                                                 strokes_pixel_indices_scnwdt,
                                                 min_stroke_length)
        
        (strokes_total_area_scnwdt, 
         strokes_avg_area_scnwdt) = calc_scnwdt_line_strokes_area_stats(
                                         strokes_long_scnwdt, 
                                         scan_widths)
        
        (strokes_select, 
         strokes_pixel_indices_select,
         scan_width_select) = select_scnwdt_line_strokes(
                                    strokes_long_scnwdt,
                                    strokes_pixel_indices_long_scnwdt,
                                    strokes_total_area_scnwdt, 
                                    strokes_avg_area_scnwdt,
                                    scan_widths)
        
        strokes_select_list.append(strokes_select)
        scan_width_select_list.append(scan_width_select)
        
        #remove pixels for the strokes selected for the minimum stroke length
        scan_image = remove_line_stroke_pixels(scan_image, 
                                               strokes_pixel_indices_select)
        
        if min_stroke_length == 1:
            #exit while loop after evaluating at a min_stroke_length of 1
            min_stroke_length_flag = False
            
        min_stroke_length = min_stroke_length/phi
        
        if min_stroke_length < 1:
            min_stroke_length = 1
    
    return strokes_select_list, scan_width_select_list

def generate_scnwdt_line_strokes(
        scan_image: np.ndarray, 
        scan_widths: list[float], 
        scan_offsets: list[float], 
        scan_angles: list[float]) -> tuple[list[list[list[list[float]]]],
                                           list[list[list[list[int]]]]]:
    '''
    Generate line strokes for each scan width line stroke set.
    
    Args:
        scan_image
        scan_widths
        scan_offsets
        scan_angles
        
    Returns: 
        strokes_scnwdt: lines strokes for each scan width line stroke
            set
        strokes_pixel_indices_scnwdt: line stroke pixel indices for each
            scan width pixel set
    '''
    
    strokes_scnwdt = []
    strokes_pixel_indices_scnwdt = []
    
    for i in range(len(scan_widths)):
        # For each scan width 
        
        temp_strokes = []
        temp_strokes_pixel_indices = []
        
        scan_width = scan_widths[i]
        
        for j in range(len(scan_offsets[i])):
            # For each scan offset 
            
            scan_offset = scan_offsets[i][j]
            
            for k in range(len(scan_angles)):
                # For each scan angle 
                
                scan_angle = scan_angles[k]
                
                (line_strokes, 
                 line_strokes_pixel_indices) = scan_line_strokes(scan_image, 
                                                                 scan_width, 
                                                                 scan_offset, 
                                                                 scan_angle)
                
                temp_strokes.append(line_strokes)
                temp_strokes_pixel_indices.append(line_strokes_pixel_indices)
                
        strokes_scnwdt.append(temp_strokes)
        strokes_pixel_indices_scnwdt.append(temp_strokes_pixel_indices)
        
    return strokes_scnwdt, strokes_pixel_indices_scnwdt

def long_pass_scnwdt_line_strokes(
        strokes_scnwdt: list[list[list[list[float]]]],
        strokes_pixel_indices_scnwdt: list[list[list[list[int]]]],
        min_stroke_length: float) -> tuple[list[list[list[list[float]]]],
                                           list[list[list[list[int]]]]]:
    '''
    Long pass line strokes for each the scan width line stroke set.
    
    Args:
        strokes_scnwdt
        strokes_pixel_indices_scnwdt
        min_strok_length
    
    Returns:
        strokes_long_scnwdt: line strokes for each line strokes 
            scan width set with strokes below minimum stroke length removed
        strokes_pixel_indices_long_scnwdt: stroke pixel indices for
            each line strokes scan width set with pixel indices removed for 
            strokes below minimum stroke length
    '''
    
    strokes_long_scnwdt = []
    strokes_pixel_indices_long_scnwdt = []
    
    for i in range(len(strokes_scnwdt)):
        # For each scan width
        
        temp_strokes = []
        temp_strokes_pixel_indices = []
        
        for j in range(len(strokes_scnwdt[i])):
            # For each strokes set of combination offset and angle
            
            (strokes_long,
             strokes_pixel_indices_long) = long_pass_line_strokes(
                                            strokes_scnwdt[i][j], 
                                            strokes_pixel_indices_scnwdt[i][j],
                                            min_stroke_length)
            
            temp_strokes.append(strokes_long)
            temp_strokes_pixel_indices.append(strokes_pixel_indices_long)
        
        strokes_long_scnwdt.append(temp_strokes)
        strokes_pixel_indices_long_scnwdt.append(temp_strokes_pixel_indices)
        
    return strokes_long_scnwdt, strokes_pixel_indices_long_scnwdt

def calc_scnwdt_line_strokes_area_stats(
        strokes_scnwdt: list[list[list[list[float]]]], 
        scan_widths: list[float]) -> tuple[list[list[float]],
                                           list[list[float]]]:
    '''
    Calculate the line stroke stats for each the scan width line stroke set.
    
    Args:
        strokes_scnwdt
        scan_widths
        
    Returns:
        strokes_total_area_scnwdt: total area of all strokes of each
            line strokes scan width set
        strokes_avg_area_scnwdt: average area of all strokes of each
            line strokes scan width set
    '''
    
    strokes_total_area_scnwdt = []
    strokes_avg_area_scnwdt = []
    
    for i in range(len(strokes_scnwdt)):
        # For each scan width
        
        temp_strokes_total_area = []
        temp_strokes_avg_area = []
        
        for j in range(len(strokes_scnwdt[i])):
            # For each strokes set of combination offset and angle
            
            strokes_length = calc_line_strokes_length(strokes_scnwdt[i][j])
            
            strokes_area = calc_line_strokes_area(strokes_length,
                                                  scan_widths[i])
            
            strokes_total_area = calc_line_strokes_total_area(
                                        strokes_area)
            
            temp_strokes_total_area.append(strokes_total_area)
            
            strokes_avg_area = calc_line_strokes_average_area(
                                        strokes_area)
            
            temp_strokes_avg_area.append(strokes_avg_area)
            
        strokes_total_area_scnwdt.append(temp_strokes_total_area)
        strokes_avg_area_scnwdt.append(temp_strokes_avg_area)
            
    return strokes_total_area_scnwdt, strokes_avg_area_scnwdt

def select_scnwdt_line_strokes(
        strokes_scnwdt: list[list[list[list[float]]]],
        strokes_pixel_indices_scnwdt: list[list[list[list[int]]]],
        strokes_total_area_scnwdt: list[list[float]], 
        strokes_avg_area_scnwdt: list[list[float]],
        scan_widths: list[float]) -> tuple[list[list[float]],
                                           list[list[int]],
                                           float]:
    
    '''
    Select line strokes for each scan width line stroke set. 
    This contains the primary criteria by which line strokes are selected 
    within and across line stroke scan width sets.
    
    Args:
        strokes_scnwdt
        strokes_pixel_indices_scnwdt
        strokes_total_area_scnwdt
        strokes_avg_area_scnwdt
        scan_widths
    
    Returns:
        strokes_select
        strokes_pixel_indices_select
        scan_width_select
    '''
    
    strokes_off_ang = []
    strokes_pixel_indices_off_ang = []
    strokes_avg_area_off_ang = []
    
    for i in range(len(strokes_scnwdt)):
        # For each scan width
        
        # For strokes of the same scan width, select the strokes with the
        # largests total area
        select_index_1 = strokes_total_area_scnwdt[i].index(
            max(strokes_total_area_scnwdt[i]))
        
        strokes_off_ang.append(strokes_scnwdt[i][select_index_1])
        strokes_pixel_indices_off_ang.append(
            strokes_pixel_indices_scnwdt[i][select_index_1])
        strokes_avg_area_off_ang.append(
            strokes_total_area_scnwdt[i][select_index_1])
    
    # Select the strokes with the largest average area across all scan widths
    # TODO: consider removing for modifying this selection criteria, 
    # larger stroke widths almost always lose out to stroke width = 1
    select_index_2 = strokes_avg_area_off_ang.index(
        max(strokes_avg_area_off_ang))
    
    strokes_select = strokes_off_ang[select_index_2]
    strokes_pixel_indices_select = strokes_pixel_indices_off_ang[
                                        select_index_2]
    scan_width_select = scan_widths[select_index_2]
            
    return strokes_select, strokes_pixel_indices_select, scan_width_select
    

#**************************************************************************
# Functions that follow are high level and used for development only

def scan_single_color_single_width_single_angle(scan_width: float,
                                                scan_offset: float,
                                                scan_angle: float):
    '''
    Line stoke scan with single scan color, single scan width, 
    and single scan angle.
        
    Args:
        scan_width
        scan_offset
        scan_angle
        
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    binary_image = cv2.imread(os.path.join(DATA_PATH, 
                                           'color_center_0_binary.png'))
    
    (line_strokes, 
     line_strokes_pixel_indices) = scan_line_strokes(binary_image, 
                                                     scan_width, 
                                                     scan_offset, 
                                                     scan_angle)
    
    image_height, image_width = binary_image.shape[:2]
    x_min = -3
    x_max = image_width + 3
    y_min = -3
    y_max = image_height + 3
    width_x = (x_max - x_min)
    width_y = (y_max - y_min)
    view_box = [x_min, y_min, width_x, width_y]
    
    write_lines_to_svg('scan_single_color', view_box, line_strokes, scan_width)
    
def scan_multiple_colors_single_width_single_angle(scan_width: float, 
                                                   scan_offset: float, 
                                                   scan_angle: float):
    
    '''
    Line stroke scan with multiple colors, single scan width, 
    and single scan angle.
    
    Args:
        scan_width
        scan_offset
        scan_angle
    
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    with open(os.path.join(DATA_PATH, 'image_color_center_bgr.txt'), 'r') as f:
        image_color_centers = json.load(f)
    f.close()
    
    color_center_line_strokes = []
    for i in range(len(image_color_centers)):
        binary_image = cv2.imread(os.path.join(DATA_PATH, 
                                  'color_center_' + str(i) + '_binary.png'))
        (line_strokes, 
         line_strokes_pixel_indices) = scan_line_strokes(binary_image,
                                                         scan_width, 
                                                         scan_offset, 
                                                         scan_angle)
        color_center_line_strokes.append(line_strokes)
    
    binary_image = cv2.imread(os.path.join(DATA_PATH, 
                                           'color_center_0_binary.png'))
    image_height, image_width = binary_image.shape[:2]
    x_min = -3
    x_max = image_width + 3
    y_min = -3
    y_max = image_height + 3
    width_x = (x_max - x_min)
    width_y = (y_max - y_min)
    view_box = [x_min, y_min, width_x, width_y]
    
    write_color_centers_line_strokes_to_svg(view_box, 
                                            image_color_centers,
                                            color_center_line_strokes, 
                                            scan_width)

def scan_single_color_large_to_small_width_single_angle(
        scan_widths: list[float],
        scan_offsets: list[float],
        scan_angle: float):
    
    '''
    Line stoke scan with a single color, large to small scan widths,
    and single scan angle.
    
    Args:
        scan_widths
        scan_offsets
        scan_angle
    
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    line_strokes_scan_widths = []
    scan_image = cv2.imread(os.path.join(DATA_PATH, 
                                         'color_center_1_binary.png'))
    
    for i in range(len(scan_widths)):
        
        (line_strokes, 
         line_strokes_pixel_indices) = scan_line_strokes(scan_image, 
                                                         scan_widths[i], 
                                                         scan_offsets[i], 
                                                         scan_angle)
        
        scan_image = remove_line_stroke_pixels(scan_image, 
                                               line_strokes_pixel_indices)
        
        cv2.imwrite(os.path.join(DATA_PATH, 'remove_line_stroke_pixels_' 
                                 + str(scan_widths[i]) + '.png'), scan_image)
            
        line_strokes_scan_widths.append(line_strokes)
    
    image_height, image_width = scan_image.shape[:2]
    x_min = -3
    x_max = image_width + 3
    y_min = -3
    y_max = image_height + 3
    width_x = (x_max - x_min)
    width_y = (y_max - y_min)
    view_box = [x_min, y_min, width_x, width_y]
    
    write_scnwdt_lines_to_svg('scan_single_color_large_to_small', 
                       view_box, line_strokes_scan_widths, scan_widths)

def scan_single_color_greedy_algorithm(
        scan_widths: list[float], 
        scan_offsets: list[float], 
        scan_angles: list[float]):
    
    '''
    Line stroke scan greedly algorithm with a single color,
    multiple scan widths, and multiple scan angles.
    
    Args:
        scan_widths
        scan_offsets
        scan_angles
    
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    scan_image = cv2.imread(os.path.join(DATA_PATH, 
                                         'color_center_1_binary.png'))
    
    (strokes_list,
     scan_width_list) = select_line_strokes(scan_image, 
                                            scan_widths, 
                                            scan_offsets, 
                                            scan_angles)
    
    image_height, image_width = scan_image.shape[:2]
    x_min = -3
    x_max = image_width + 3
    y_min = -3
    y_max = image_height + 3
    width_x = (x_max - x_min)
    width_y = (y_max - y_min)
    view_box = [x_min, y_min, width_x, width_y]
    
    write_scnwdt_lines_to_svg('single_greedy', view_box, 
                              strokes_list, scan_width_list)

def scan_multiple_colors_greedy_algorithm(scan_widths: list[float], 
                                          scan_offsets: list[float], 
                                          scan_angles: list[float]):
    
    '''
    Line stroke scan greedly algorithm with a multiple colors,
    multiple scan widths, and multiple scan angles.
    
    Args:
        scan_widths
        scan_offsets
        scan_angles
    
    Returns:
        Outputs a svg image to the data folder.
    '''
    
    with open(os.path.join(DATA_PATH, 'image_color_center_bgr.txt'), 'r') as f:
        image_color_centers = json.load(f)
    f.close()
    
    color_centers_strokes_list = []
    color_centers_scan_width_list = []
    
    for i in range(len(image_color_centers)):
            
        scan_image = cv2.imread(os.path.join(DATA_PATH, 
                                  'color_center_' + str(i) + '_binary.png'))
        
        (strokes_list,
         scan_width_list) = select_line_strokes(scan_image, 
                                                scan_widths, 
                                                scan_offsets, 
                                                scan_angles)
        
        color_centers_strokes_list.append(strokes_list)
        color_centers_scan_width_list.append(scan_width_list)
    
    binary_image = cv2.imread(os.path.join(DATA_PATH, 
                                           'color_center_0_binary.png'))
    image_height, image_width = binary_image.shape[:2]
    x_min = -3
    x_max = image_width + 3
    y_min = -3
    y_max = image_height + 3
    width_x = (x_max - x_min)
    width_y = (y_max - y_min)
    view_box = [x_min, y_min, width_x, width_y]
    
    write_color_centers_scnwdt_strokes_to_svg('multiple_greedy',
                                              view_box, 
                                              image_color_centers,
                                              color_centers_strokes_list, 
                                              color_centers_scan_width_list)

if __name__ == '__main__':
    
    # scan_width = 1
    # scan_offset = -0.5
    # scan_angle = 0
    
    # scan_single_color_single_width_single_angle(scan_width,
    #                                             scan_offset,
    #                                             scan_angle)
    
    # scan_multiple_colors_single_width_single_angle(scan_width, 
    #                                                scan_offset, 
    #                                                scan_angle)
    
    # scan_widths = [5, 4, 3, 2, 1]
    # scan_offsets = [-2, -1.5, -1, -0.5, 0]
    # scan_angle = 45
    
    # scan_single_color_large_to_small_width_single_angle(scan_widths,
    #                                                     scan_offsets,
    #                                                     scan_angle)
    
    scan_widths = [1.0]
    
    scan_offsets = []
    for i in range(len(scan_widths)):
        scan_offsets.append(calc_nested_scan_offsets(scan_widths[i]))
    
    # scan_angles = [i for i in range(0, 180, 15)]
    scan_angles = [45, 135]
    
    # scan_single_color_greedy_algorithm(scan_widths, 
    #                                    scan_offsets, 
    #                                    scan_angles)
    
    scan_multiple_colors_greedy_algorithm(scan_widths, 
                                          scan_offsets, 
                                          scan_angles)