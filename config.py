import numpy as np
import math

# set variables for quasi-memorization mechanism
SAVED_LB, SAVED_LT, SAVED_RB, SAVED_RT = None, None, None, None
ACTIVE = False
learn_rate = 0.5

# set variables for roi
bot_pad = 1/13 # bottom left and right padding
top_pad = 6/13 # top left and right padding
h_pad = 3/5 # height padding

# set variables for gaussian and canny
kernel_size = 5 # define kernel size for gaussian filter
low_threshold = 50 # low threshold for canny
high_threshold = 150 # high threshold for canny

# set variables for hough transformation
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 30     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 80 # minimum number of pixels making up a line
max_line_gap = 160    # maximum gap in pixels between connectable line segments

# set threshold for color white and yellow
white_lo = [100, 100, 200]
white_hi = [255, 255, 255]
yellow_lo = [20, 120, 80]
yellow_hi = [45, 200, 255]