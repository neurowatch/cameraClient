import cv2
import settings

class GetMOG2BackgroundSubstractorFrame:

    def __init__(self, history) -> None:
        self.history = history
        self.backgroundSub = cv2.createBackgroundSubtractorMOG2()

    def execute(self, cap):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            background_frame, frame

        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)
            
        background_frame = self.backgroundSub.apply(frame1)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_GRAY2BGR)

        return background_frame, frame