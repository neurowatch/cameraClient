import cv2

class CreateClip:
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
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.total_frames = self.duration * self.frame_rate
        self.current_frame = 0

    def execute(self, frame):
        if self.clip == None:
            height, width, _ = frame.shape
            self.create_clip(height, width)

        self.clip.write(frame)
        self.current_frame += 1

    def is_completed(self):
        if self.current_frame < self.total_frames:
            return False
        else:
            return True

    def create_clip(self, height, width):
        #ffmpeg -i input.mp4 -vcodec h264 -acodec aac output.mp4
        self.clip = cv2.VideoWriter(self.file_name, self.fourcc, self.frame_rate, (width, height))

    def on_complete(self):
        self.clip.release()
        self.clip = None
        return self.file_name