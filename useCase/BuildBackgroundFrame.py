import cv2
import numpy as np
import settings

class BuildBackgroundFrame:

    def __init__(self, history) -> None:
        self.history = history

    def execute(self, cap):
        background_frames = []
        for _ in range(self.history):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                exit()
            frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
            frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

            background_frames.append(frame1)

        background_frame = BuildBackgroundFrame.get_median_background_frame(background_frames)
        
        return background_frame, frame

    @staticmethod
    def get_median_background_frame(frames):
        return np.median(np.array(frames), axis=0).astype(np.uint8)