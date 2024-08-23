from CameraController import CameraController
from useCase import GetMOG2BackgroundSubstractorFrame, DetectMotion
import time

def test_detection():
    start_time = time.time()
    cameraController = CameraController(
        upload_clip=False,
        show=True,
    )
    frame_count = cameraController.caputre_video()
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

test_detection()