import motion_detection
import time

def test_background_substraction():
    start_time = time.time()
    motionDetection = motion_detection.MotionDetection()
    frame_count = motionDetection.caputre_video()
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

test_background_substraction()