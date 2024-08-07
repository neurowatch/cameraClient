import os
import cv2
import csv
import time
from useCase import BuildBackgroundFrame, DetectMotion, GetMOG2BackgroundSubstractorFrame, DetectObjects
from CameraController import CameraController
import settings

class TestBed:

    def __init__(self) -> None:
        self.start_time = -1
        self.end_time = -1
        self.root = os.path.abspath(os.getcwd())
        self.create_results_file()

    def run_testbed(self):
        self.test_controller()
        self.test_background_substraction_methods()
        self.test_motion_detection()
        self.test_object_detection()

    def test_controller(self):
        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        
        cameraController = CameraController(upload_clip=False, show=False, source=input)
        frame_count = cameraController.caputre_video()
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test CameraController",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def test_background_substraction_methods(self):
        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        useCase = BuildBackgroundFrame(10)
        useCase.execute(cap)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test custom background substraction BuildBackgroundFrame",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

        cap.release()

        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        useCase = GetMOG2BackgroundSubstractorFrame(10)
        useCase.execute(cap)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test cv2 background substraction GetMOG2BackgroundSubstractorFrame",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def test_motion_detection(self):
        background_frame_usecase = BuildBackgroundFrame(10)
        detect_motion_use_case = DetectMotion(background_frame_usecase)

        input = os.path.join(self.root, "testbed/sources/input1.mp4")
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        background_frame, frame = background_frame_usecase.execute(cap)

        frame1 = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
        frame1 = cv2.GaussianBlur(frame1, settings.BLUR_KERNEL, cv2.BORDER_DEFAULT)
        detect_motion_use_case.execute(cap, background_frame, frame1)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test motion detection",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def test_object_detection(self):
        input = os.path.join(self.root, "testbed/sources/input1.mp4")

        detect_objects_use_case = DetectObjects(use_ncnn=False)
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detect_objects_use_case.execute(frame)
            current_frame += 1

        cap.release()
        cv2.destroyAllWindows()

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test YOLOv8 model",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

        detect_objects_use_case = DetectObjects(use_ncnn=True)
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detect_objects_use_case.execute(frame)
            current_frame += 1

        cap.release()
        cv2.destroyAllWindows()

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test YOLOv8 ncnn model",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def create_results_file(self):
        with open('results.csv', mode='w', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(["Test Name", "Frame Count", "Total Time", "FPS"])

    def write_results(self, test_name, frame_count, total_time, fps):
        print(f"Test: {test_name}. Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")
        with open('results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow([test_name, frame_count, total_time, fps])