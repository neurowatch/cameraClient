import cv2
import numpy as np
import settings
from .DetectLigthChange import DetectLightChange

class DetectMotion:
    
    def __init__(self, background_frame_usecase):
        self.background_frame_usecase = background_frame_usecase
        self.frame_counter = 0

    def execute(self, cap, background_frame, frame1):
        # Frames for frame differencing.
        frame = cap.read()[1]
        frame2 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame2 = cv2.GaussianBlur(frame2, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        if (self.frame_counter % settings.UPDATE_INTERVAL == 0) or DetectLightChange.execute(background_frame, frame2) :
            background_frame, frame = self.background_frame_usecase.execute(cap)
            self.frame_counter = 0

        self.frame_counter += 1

        binary_max_diff = self.process_three_frame_differencing(background_frame, frame1, frame2)

        motion_detected = np.any(binary_max_diff > 0)

        return motion_detected, frame, frame2, background_frame
    
    def process_three_frame_differencing(self, frame1, frame2, frame3):
        
        print(frame1.shape, frame2.shape, frame3.shape)
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)

        max_diff = cv2.max(diff1, diff2)
        gray_max_diff = cv2.cvtColor(max_diff, cv2.COLOR_BGR2GRAY)
        _, binary_max_diff = cv2.threshold(gray_max_diff, 25, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_CLOSE, kernel)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_OPEN, kernel)

        return binary_max_diff