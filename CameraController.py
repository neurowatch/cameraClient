import cv2
import settings
from service import NeurowatchService
from useCase import BuildBackgroundFrame, CreateClip, DetectMotion, DetectObjects, SaveClip, DetectedObjectsStore

class CameraController:

    def __init__(
            self, 
            upload_clip=True, 
            show=False, 
            source=settings.SOURCE,
            detect_motion_usecase = DetectMotion(),
            detect_object_usecase = DetectObjects(),
            create_clip_usecase = CreateClip(),
            save_clip_usecase = SaveClip(NeurowatchService()),
            detected_objects_store_usecase = DetectedObjectsStore()
        ):
        
        self.detect_motion_use_case = detect_motion_usecase
        self.detect_object_use_case = detect_object_usecase
        self.create_clip_use_case = create_clip_usecase
        self.save_clip_use_case = save_clip_usecase
        self.detected_objects_store_use_case = detected_objects_store_usecase
        self.source = source
        self.upload_clip = upload_clip
        self.show = show

    def caputre_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Could not open the video")
            exit()

        background_frame, frame = BuildBackgroundFrame.execute(cap, settings.HISTORY)

        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        create_clip = False
        frame_counter = 0
        current_clip_frame = 0

        while True:
            motion_detected, frame_raw, frame2_processed, background_frame = self.detect_motion_use_case.execute(cap, background_frame, frame1)
            if (motion_detected):
                detected_objects_in_frame = self.detect_object_use_case.execute(frame_raw)
                self.detected_objects_store_use_case.store(detected_objects_in_frame, current_clip_frame)
                create_clip = True

            if create_clip == True:
                current_clip_frame = self.create_clip_use_case.execute(frame_raw)
                if self.create_clip_use_case.is_completed():
                    if (self.upload_clip):
                        self.save_clip_use_case.execute(
                            file_path = self.create_clip_use_case.on_complete(),
                            detected_objects = self.detected_objects_store_use_case.pop()
                        )
                    else:
                        self.create_clip_use_case.on_complete()

                    create_clip = False
                    current_clip_frame = 0
                    break

            frame1 = frame2_processed
            
            frame_counter += 1
            
            if self.show:
                frames_to_combine = [background_frame, frame1]
                combined_frame = cv2.hconcat(frames_to_combine)
                cv2.imshow("Frames", combined_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return frame_counter