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
            background_substraction_usecase = BuildBackgroundFrame(settings.HISTORY),
            detect_motion_usecase = DetectMotion(BuildBackgroundFrame(settings.HISTORY)),
            detect_object_usecase = DetectObjects(),
            create_clip_usecase = CreateClip(),
            save_clip_usecase = SaveClip(NeurowatchService()),
            detected_objects_store_usecase = DetectedObjectsStore()
        ):
        
        self.background_substraction_use_case = background_substraction_usecase
        self.detect_motion_use_case = detect_motion_usecase
        self.detect_object_use_case = detect_object_usecase
        self.create_clip_use_case = create_clip_usecase
        self.save_clip_use_case = save_clip_usecase
        self.detected_objects_store_use_case = detected_objects_store_usecase
        self.source = source
        self.upload_clip = upload_clip
        self.show = show

    def caputre_video(self):
        # Obtains the VideoCapture object from the selected source
        cap = cv2.VideoCapture(self.source)

        # If opening the source is not possible, show an error and exit
        if not cap.isOpened():
            print("Error: Could not open the video")
            exit()

        # Obtains the background frame from the use case, it also returns the current frame
        background_frame, frame = self.background_substraction_use_case.execute(cap)

        # Preprocess the frame
        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        # Flag to determine if a clip should be created
        create_clip = False
        # Keeps a count of the frames for logging purposes
        frame_counter = 0
        # Keeps a count of the frames in the clip
        current_clip_frame = 0

        while True:
            # Checks if motion is detected, also returns other relevant data such as the current frame, raw and preprocessed and the current background frame
            # which may have been updated
            motion_detected, frame_raw, frame2_processed, background_frame = self.detect_motion_use_case.execute(cap, background_frame, frame1)
            if (motion_detected):
                # Obtains any objects detected in the frame, the unprocessed frame is passed
                detected_objects_in_frame = self.detect_object_use_case.execute(frame_raw)
                # Checks if the list is not empty
                if detected_objects_in_frame:
                    # Stores the objects, also which frame in the clip it has been detected
                    self.detected_objects_store_use_case.store(detected_objects_in_frame, current_clip_frame)
                    # Sets the flag to True
                    create_clip = True

            if create_clip == True:
                # Creates the clip file and adds and returns the current_clip_frame, this clip value is actually used in the next iteration
                # for that reason is actually the current clip frame + 1
                current_clip_frame = self.create_clip_use_case.execute(frame_raw)
                # Checks if the clip creation is completed
                if self.create_clip_use_case.is_completed():
                    # If the upload clip flag is True, will start uploading the clip, obtains the file and the detected objects.
                    if (self.upload_clip):
                        self.save_clip_use_case.execute(
                            file_path = self.create_clip_use_case.on_complete(),
                            detected_objects = self.detected_objects_store_use_case.pop()
                        )
                    else:
                        # Otherwise just calls on_complete 
                        self.create_clip_use_case.on_complete()

                    # Resets flag and counter.
                    create_clip = False
                    current_clip_frame = 0

            # Assigns the current frame to the previous, this is used in the following loop
            if frame2_processed is not None:
                frame1 = frame2_processed        
                frame_counter += 1
            else:
                break
            
            # If show is True, shows the background frame and frame1
            if self.show:
                frames_to_combine = [background_frame, frame1]
                combined_frame = cv2.hconcat(frames_to_combine)
                cv2.imshow("Frames", combined_frame)

            # Ends execution
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        return frame_counter