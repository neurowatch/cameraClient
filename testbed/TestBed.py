import os
import cv2
import csv
import time
from useCase import BuildBackgroundFrame, DetectMotion, GetMOG2BackgroundSubstractorFrame, DetectObjects, DetectObjectsSSDLite, DetectObjectsSSD, DetectedObjectsStore, DetectObjectsTrack
from CameraController import CameraController
import settings

class TestBed:

    def __init__(self) -> None:
        self.start_time = -1
        self.end_time = -1
        self.root = os.path.abspath(os.getcwd())
        self.create_results_file()
        self.sources = [
            {
                "source": "testbed/sources/test_clip_1.mp4",
                "expected_objects": 0
            },
            {
                "source": "testbed/sources/test_clip_2.mp4",
                "expected_objects": 4
            },
            {
                "source": "testbed/sources/test_clip_3.mp4",
                "expected_objects": 0
            },
            {
                "source": "testbed/sources/test_clip_4.mp4",
                "expected_objects": 4
            },
            {
                "source": "testbed/sources/test_clip_5.mp4",
                "expected_objects": 3
            },

        ]

    def run_testbed(self):
        for source in self.sources:
            self.write_subtitle(source["source"])
            self.test_controller(source["source"])
            self.test_background_substraction_methods(source["source"])
            self.test_motion_detection(source["source"])
            self.test_object_detection_yolo(source["source"])
            self.test_object_detection_yolo_track(source["source"])
            self.test_object_detection_torchvision_ssd(source["source"])
            self.test_object_detection_torchvision_ssdlite(source["source"])
            self.write_detection_headers()
            self.test_object_detection_accuracy_yolo(source["source"], source["expected_objects"])
            self.test_object_detection_accuracy_yolo_track(source["source"], source["expected_objects"])
            self.test_object_detection_accuracy_torchvision(source["source"], source["expected_objects"])
            self.test_object_detection_accuracy_torchvision_lite(source["source"], source["expected_objects"])            

    def test_controller(self, source):
        input = os.path.join(self.root, source)
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

    def test_background_substraction_methods(self, source):
        input = os.path.join(self.root, source)
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

    def test_motion_detection(self, source):
        background_frame_usecase = BuildBackgroundFrame(10)
        detect_motion_use_case = DetectMotion(background_frame_usecase)

        input = os.path.join(self.root, source)
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

    def test_object_detection_yolo(self, source):
        input = os.path.join(self.root, source)

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

    def test_object_detection_yolo_track(self, source):
        input = os.path.join(self.root, source)

        detect_objects_use_case = DetectObjectsTrack(use_ncnn=False)
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
            test_name="Test YOLOv8 track model",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

        detect_objects_use_case = DetectObjectsTrack(use_ncnn=True)
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
            test_name="Test YOLOv8 track ncnn model",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def test_object_detection_torchvision_ssdlite(self, source):
        input = os.path.join(self.root, source)

        detect_objects_use_case = DetectObjectsSSDLite()
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detect_objects_use_case.execute(frame)
            current_frame += 1

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test ssdlite320_mobilenet_v3_large object detection",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )

    def test_object_detection_torchvision_ssd(self, source):
        input = os.path.join(self.root, source)

        detect_objects_use_case = DetectObjectsSSD()
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detect_objects_use_case.execute(frame)
            current_frame += 1

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        fps = frame_count / total_time
        self.write_results(
            test_name="Test ssd300_vgg16 object detection",
            frame_count=frame_count,
            total_time=total_time,
            fps=fps
        )


    def test_object_detection_accuracy_yolo(self, source, expected_count):
        input = os.path.join(self.root, source)

        detected_objects_store_use_case = DetectedObjectsStore()

        detect_objects_use_case = DetectObjects(use_ncnn=False)
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test YOLOv8 model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

        detect_objects_use_case = DetectObjects(use_ncnn=True)
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test YOLOv8 ncnn model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

    def test_object_detection_accuracy_torchvision(self, source, expected_count):
        input = os.path.join(self.root, source)

        detected_objects_store_use_case = DetectedObjectsStore()
        detect_objects_use_case = DetectObjectsSSD()
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test TorchVision SSD model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

    def test_object_detection_accuracy_torchvision_lite(self, source, expected_count):
        input = os.path.join(self.root, source)

        detected_objects_store_use_case = DetectedObjectsStore()
        detect_objects_use_case = DetectObjectsSSDLite()
        self.start_time = time.time()
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test TorchVision SSDLite model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

    def test_object_detection_accuracy_yolo_track(self, source, expected_count):
        input = os.path.join(self.root, source)

        detected_objects_store_use_case = DetectedObjectsStore()

        detect_objects_use_case = DetectObjectsTrack(use_ncnn=False)
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test YOLOv8 track settings model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

        detect_objects_use_case = DetectObjectsTrack(use_ncnn=True)
        cap = cv2.VideoCapture(input)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while current_frame < frame_count:
            frame = cap.read()[1]
            detected_objects_in_frame = detect_objects_use_case.execute(frame)
            if detected_objects_in_frame:
                detected_objects_store_use_case.store(detected_objects_in_frame, current_frame)
            current_frame += 1

        detected_objects = detected_objects_store_use_case.pop()
        cap.release()
        cv2.destroyAllWindows()

        self.write_accuracy_results(
            test_name="Test YOLOv8 track ncnn model accuracy",
            detected_objects=len(detected_objects),
            expected_objects=expected_count
        )

    def create_results_file(self):
        with open('results.csv', mode='w', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(["Test Name", "Frame Count", "Total Time", "FPS"])

    def write_detection_headers(self):
        with open('results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(["Test Name", "Detected objects", "Expected objects"])

    def write_results(self, test_name, frame_count, total_time, fps):
        print(f"Test: {test_name}. Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")
        with open('results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow([test_name, frame_count, total_time, fps])

    def write_subtitle(self, soruce):
        with open('results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow([soruce])

    def write_accuracy_results(self, test_name, detected_objects, expected_objects):
        print(f"Test: {test_name}. Detected objects: {detected_objects}, Expected objets: {expected_objects}")
        with open('results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow([test_name, detected_objects, expected_objects])
