import cv2
import settings
from useCase import BuildBackgroundFrame
from useCase import CreateClip
from useCase import DetectMotion
from useCase import DetectObjects

class CameraController:

    def __init__(self):
        self.detectMotionUseCase = DetectMotion()
        self.detectObjectUseCase = DetectObjects()
        self.createClipUseCase = CreateClip()

    def caputre_video(self):
        cap = cv2.VideoCapture(settings.SOURCE)
        if not cap.isOpened():
            print("Error: Could not open the video")
            exit()

        background_frame, frame = BuildBackgroundFrame.execute(cap, settings.HISTORY)

        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        create_clip = False
        frame_counter = 0

        while True:
            motion_detected, frame2, frame2_processed = self.detectMotionUseCase.execute(cap, background_frame, frame1)
            if (motion_detected):
                detectedObjects = self.detectObjectUseCase.execute(frame2)
                # TODO: Store detected objects
                create_clip = True

            if create_clip == True:
                self.createClipUseCase.execute(frame2)

            if self.createClipUseCase.is_completed():
                self.createClipUseCase.on_complete()
                break
                #Upload the clip

            frame1 = frame2_processed
            
            frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return frame_counter