from lane_line import *
from config import *
from moviepy.editor import VideoFileClip
import os

if __name__ == '__main__':
    global SAVED_LB, SAVED_LT, SAVED_RB, SAVED_RT, ACTIVE
    SAVED_LB, SAVED_LT, SAVED_RB, SAVED_RT = None, None, None, None
    ACTIVE = False
    white_output = 'output/input2.mp4'
    clip1 = VideoFileClip("source/input2.mp4")
    white_clip = clip1.fl_image(process_image)
    get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')