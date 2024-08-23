import cv2
import numpy as np
import settings
from .DetectLigthChange import DetectLightChange

class DetectMotion:
    
    def __init__(self, background_frame_usecase):
        self.background_frame_usecase = background_frame_usecase
        self.frame_counter = 0

    def execute(self, cap, background_frame, frame1):
        ret, frame = cap.read()
        if not ret:
            print("No frame to read")
            return False, None, None, None

        # Obtains the second frame for frame differencing        
        frame2 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame2 = cv2.GaussianBlur(frame2, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        # Checks if a new background frame is required
        if (self.frame_counter % settings.UPDATE_INTERVAL == 0) or DetectLightChange.execute(background_frame, frame2) :
            background_frame, frame = self.background_frame_usecase.execute(cap, background_frame)
            self.frame_counter = 0

        self.frame_counter += 1
        
        # Obtains the binary max diff of the bg frame and the two succeding frames
        binary_max_diff = self.process_three_frame_differencing(background_frame, frame1, frame2)

        # If the binary diff is greater than 0, motion is detected
        motion_detected = np.any(binary_max_diff > 0)

        return motion_detected, frame, frame2, background_frame
    
    def process_three_frame_differencing(self, frame1, frame2, frame3):

        # First obtains the diff between the frames        
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)

        # Obtains the max pixel values of the differenciated frames
        max_diff = cv2.max(diff1, diff2)
        gray_max_diff = cv2.cvtColor(max_diff, cv2.COLOR_BGR2GRAY)

        # Applies a threshold function
        _, binary_max_diff = cv2.threshold(gray_max_diff, 25, 255, cv2.THRESH_BINARY)

        # Some processing
        kernel = np.ones((5, 5), np.uint8)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_CLOSE, kernel)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_OPEN, kernel)

        # Returns the binary max diff
        return binary_max_diff