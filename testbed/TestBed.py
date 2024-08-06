import os
import cv2
import time
from useCase import BuildBackgroundFrame, DetectMotion
from CameraController import CameraController
import settings

class TestBed:

    def __init__(self) -> None:
        self.start_time = -1
        self.end_time = -1
        self.root = os.path.abspath(os.getcwd())

    def test_controller(self):
        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        
        cameraController = CameraController(upload_clip=False, show=False, source=input)
        frame_count = cameraController.caputre_video()
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

    def test_background_substraction_methods(self):
        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        BuildBackgroundFrame.execute(cap, 10)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

    def test_motion_detection(self):
        detect_motion_use_case = DetectMotion()
        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        background_frame, frame = BuildBackgroundFrame.execute(cap, 10)
        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)
        detect_motion_use_case.execute(cap, background_frame, frame1)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")