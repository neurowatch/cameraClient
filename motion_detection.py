import cv2
import numpy as np
import math
from ultralytics import YOLO

class MotionDetection:

    RESIZE_WIDTH = 320
    RESIZE_HEIGHT = 240
    BLUR_KERNEL = (5, 5)

    def __init__(self, source=0, history=10):
        self.source = source
        self.history = history
        self.model = YOLO("yolov8n.pt")

        #self.background_substractor = cv2.createBackgroundSubtractorMOG2(history=self.history)

    def caputre_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Could not open the video")
            exit()

        # Read inital frame to get video dimensions
        #ret, frame = cap.read()
        #if not ret:
        #    print("Error: Could not read initial frame")
        #    exit()


        background_frames = []
        for _ in range(self.history):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                exit()
            frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
            frame = cv2.GaussianBlur(frame, self.BLUR_KERNEL, cv2.BORDER_DEFAULT)

            background_frames.append(frame)

        background_frame = self.get_median_background_frame(background_frames)
        background_frames = []

        frame_counter = 0
        update_interval = 100

        frame1 = cap.read()[1]
        frame1 = cv2.resize(frame1, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, self.BLUR_KERNEL, cv2.BORDER_DEFAULT)

        while True:
            #ret, frame = cap.read()

            # Frames for frame differencing.
            frame2 = cap.read()[1]
            frame2 = cv2.resize(frame2, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
            resized_frame1 = frame2
            frame2 = cv2.GaussianBlur(frame2, self.BLUR_KERNEL, cv2.BORDER_DEFAULT)

            if frame_counter % update_interval == 0:
                background_frames = []
                for _ in range(self.history):
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
                    frame = cv2.GaussianBlur(frame, self.BLUR_KERNEL, cv2.BORDER_DEFAULT)
                    if not ret: 
                        break
                    background_frames.append(frame)
                
                if len(background_frames) == self.history:
                    background_frame = self.get_median_background_frame(background_frames)

            frame_counter += 1

            binary_max_diff, frame_max_diff = self.process_three_frame_differencing(background_frame, frame1, frame2)

            motion_detected = np.any(binary_max_diff > 0)

            frames_to_combine = [frame_max_diff, background_frame]
            if (motion_detected):
                print(f"motion detected! in frame: {frame_counter}")

                detected_objects = self.detect_objects(resized_frame1)
                frames_to_combine.append(detected_objects)

            combined_frame = cv2.hconcat(frames_to_combine)

            cv2.imshow("Live and background", combined_frame)

            frame1 = frame2

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return frame_counter
            

    def update_background_frame(self, background_frame):
        background_frame = self.background_substractor.getBackgroundImage()        
        return background_frame
    
    def get_median_background_frame(self, frames):
        return np.median(np.array(frames), axis=0).astype(np.uint8)


    def process_three_frame_differencing(self, frame1, frame2, frame3):
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)

        max_diff = cv2.max(diff1, diff2)
        gray_max_diff = cv2.cvtColor(max_diff, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.uint8)
        _, binary_max_diff = cv2.threshold(gray_max_diff, 25, 255, cv2.THRESH_BINARY)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_CLOSE, kernel)
        binary_max_diff = cv2.morphologyEx(binary_max_diff, cv2.MORPH_OPEN, kernel)

        combined_max_diff = cv2.cvtColor(binary_max_diff, cv2.COLOR_GRAY2BGR)
        
        return binary_max_diff, combined_max_diff

    def detect_objects(self, frame):
        results = self.model(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0]*100))/100
                print("confidence: ", confidence)

                objectClass = int(box.cls[0])
                print("class: ", confidence)

                origin = [x1, y1]
                font = cv2.FONT_HERSHEY_PLAIN
                fontScale = 2
                color = (0, 255, 255)
                thickness = 2

                cv2.putText(frame, self.model.names[objectClass], origin, font, fontScale, color, thickness)

        return frame

    def get_contours(self, frame, fgMask):
        contours = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rois = []

        for contour in contours:
            #Ignore small contours
            if cv2.contourArea(contour < 500):
                continue 

            (x, y, w, h) = cv2.boundingRect(contour)
            rois.append(frame[y:y+h, x:x+w])

        return rois