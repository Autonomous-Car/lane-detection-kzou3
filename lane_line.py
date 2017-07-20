from config import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

"""apply grayscale transform"""
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

"""apply canny transform"""
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

"""apply gaussian blur"""
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

"""apply a trapezoid region of interest bounded by the vertices"""
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

"""given a slope-intercept form of a line and y-coordinate, return x-coordinate"""
def get_x(slope, intercept, y):
    return int((y - intercept)/slope)

"""given a slope-intercept form of a line and x-coordinate, return y-coordinate"""
def get_y(slope, intercept, x):
    return int(x*slope + intercept)

"""given scattered points obtained from hough transform, approximate and plot line(s)"""
def draw_lines(img, lines, roi_height, color=[255, 0, 0], thickness=10):
    global SAVED_LB, SAVED_LT, SAVED_RB, SAVED_RT, ACTIVE
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    height, width = img.shape[0], img.shape[1]
    top_y, bot_y = int(roi_height*height), height

    # if fail to detect lanes on a specific frame
    if lines is None:
        cv2.line(img, (SAVED_LB, bot_y),(SAVED_LT, top_y), color, thickness)
        cv2.line(img, (SAVED_RB, bot_y),(SAVED_RT, top_y), color, thickness)
        return;

    # classify points into two catogories
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 < width/2: # if point in left half-plane
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif x1 >= width/2: # if point in right half-plane
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    if len(left_x) <= 1 or len(right_x) <= 1:
        if ACTIVE:
            cv2.line(img, (SAVED_LB, bot_y),(SAVED_LT, top_y), color, thickness)
            cv2.line(img, (SAVED_RB, bot_y),(SAVED_RT, top_y), color, thickness)
        return;

    # obtain linear regression on left side
    left_slope, left_int = np.polyfit(left_x, left_y, 1)
    left_bot_x = get_x(left_slope, left_int, bot_y)
    left_top_x = get_x(left_slope, left_int, top_y)

    # obtain linear regression on right side
    right_slope, right_int = np.polyfit(right_x, right_y, 1)
    right_bot_x = get_x(right_slope, right_int, bot_y)
    right_top_x = get_x(right_slope, right_int, top_y)

    # render pivot points
    if ACTIVE:
        SAVED_LB = int(0.5*left_bot_x + 0.5*SAVED_LB)
        SAVED_LT = int(0.5*left_top_x + 0.5*SAVED_LT)
        SAVED_RB = int(0.5*right_bot_x + 0.5*SAVED_RB)
        SAVED_RT = int(0.5*right_top_x + 0.5*SAVED_RT)
    else:
        SAVED_LB = left_bot_x
        SAVED_LT = left_top_x
        SAVED_RB = right_bot_x
        SAVED_RT = right_top_x
        ACTIVE = True

    # plot the lines
    cv2.line(img, (SAVED_LB, bot_y),(SAVED_LT, top_y), color, thickness)
    cv2.line(img, (SAVED_RB, bot_y),(SAVED_RT, top_y), color, thickness)

"""return an image with hough lines overlay"""
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, roi_height):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, roi_height)
    return line_img

"""add overlay image to the original image"""
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

"""filter out the color defined by the low threshold and high threshold"""
def color_filter(img, lo_list, hi_list):
    return (img[:,:,0] < lo_list[0]) | (img[:,:,0] > hi_list[0]) | \
    (img[:,:,1] < lo_list[1]) | (img[:,:,1] > hi_list[1]) | \
    (img[:,:,2] < lo_list[2]) | (img[:,:,2] > hi_list[2])

"""process a single image frame to find the lane(s)"""
def process_image(image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # select only white and yellow color
    color_layer = np.copy(image)
    hls_layer = cv2.cvtColor(color_layer, cv2.COLOR_RGB2HLS)
    white_filter = color_filter(color_layer, white_lo, white_hi)
    yellow_filter = color_filter(hls_layer, yellow_lo, yellow_hi)
    color_layer[yellow_filter & white_filter] = [0,0,0]

    # obtain the canny edges
    gray_layer = grayscale(color_layer)
    gray_layer = gaussian_blur(gray_layer, kernel_size)
    edges = canny(gray_layer, low_threshold, high_threshold)

    # obtain roi
    vertices = np.array([[(xsize*bot_pad, ysize), \
    (xsize*top_pad, ysize*h_pad), \
    (xsize*(1-top_pad), ysize*h_pad), \
    (xsize*(1-bot_pad),ysize)]], dtype=np.int32)
    edges = region_of_interest(edges, vertices)

    # add hough transform
    lines = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap, h_pad)
    result = weighted_img(lines, image, 1, 1)
    return result
