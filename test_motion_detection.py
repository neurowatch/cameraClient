from CameraController import CameraController
from useCase import GetMOG2BackgroundSubstractorFrame, DetectMotion
import time

def test_detection():
    start_time = time.time()
    background_substraction_usecase=GetMOG2BackgroundSubstractorFrame(history=10)
    cameraController = CameraController(
        upload_clip=True, 
        background_substraction_usecase=background_substraction_usecase,
        detect_motion_usecase=DetectMotion(background_substraction_usecase)
    )
    frame_count = cameraController.caputre_video()
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

test_detection()