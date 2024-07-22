import cv2
import settings
from service import NeurowatchService
from useCase import BuildBackgroundFrame, CreateClip, DetectMotion, DetectObjects, SaveClip, DetectedObjectsStore

class CameraController:

    def __init__(self):
        self.detect_motion_use_case = DetectMotion()
        self.detect_object_use_case = DetectObjects()
        self.create_clip_use_case = CreateClip()
        self.save_clip_use_case = SaveClip(NeurowatchService())
        self.detected_objects_store_use_case = DetectedObjectsStore()

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
            motion_detected, frame2, frame2_processed = self.detect_motion_use_case.execute(cap, background_frame, frame1)
            if (motion_detected):
                detected_objects_in_frame = self.detect_object_use_case.execute(frame2)
                self.detected_objects_store_use_case.store(detected_objects_in_frame)
                create_clip = True

            if create_clip == True:
                self.create_clip_use_case.execute(frame2)

            if self.create_clip_use_case.is_completed():
                self.save_clip_use_case.execute(
                    file_path = self.create_clip_use_case.on_complete(),
                    detected_objects = self.detected_objects_store_use_case.pop()
                )
                break

            frame1 = frame2_processed
            
            frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return frame_counter