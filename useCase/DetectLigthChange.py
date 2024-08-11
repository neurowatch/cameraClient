import cv2
import numpy as np

class DetectLightChange:
    '''
        Uses absdiff to check if a large ligth change has occured between two frames.
    '''
    @staticmethod
    def execute(frame1, frame2, threshold=75):
        frame1_grayscale = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_grayscale = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(frame1_grayscale, frame2_grayscale)
        mean_diff = np.mean(diff)
        return mean_diff > threshold