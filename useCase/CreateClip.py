import cv2

class CreateClip:
    '''
        Creates a video clip based on the input params. By default of 20 seconds duration and 15 fps frame rate.
    '''
    def __init__(
            self, 
            duration=20, 
            frame_rate=15,
            file_name="output.mp4"
        ) -> None:
        self.duration = duration
        self.frame_rate = frame_rate
        self.clip = None
        self.file_name = file_name
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.total_frames = self.duration * self.frame_rate
        self.current_frame = 0

    def execute(self, frame):
        if self.clip == None:
            height, width, _ = frame.shape
            self.create_clip(height, width)

        self.clip.write(frame)
        self.current_frame += 1
        return self.current_frame

    def is_completed(self):
        # Checks if the clip is created
        if self.current_frame <= self.total_frames:
            return False
        else:
            return True

    def create_clip(self, height, width):
        self.clip = cv2.VideoWriter(self.file_name, self.fourcc, self.frame_rate, (width, height))

    def on_complete(self):
        # Called after completion, releases the clip object, clears the values and returns the file name
        self.clip.release()
        self.clip = None
        self.current_frame = 0
        return self.file_name