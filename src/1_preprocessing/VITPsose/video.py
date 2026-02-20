import cv2

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        self.movie_info = {"frame_width" : int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           "frame_height" : int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           "frame_fps" : float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0}
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H264 or mp4v
        self.writer = cv2.VideoWriter(output_path, 
                                      fourcc, 
                                      self.movie_info["frame_fps"], 
                                      (self.movie_info["frame_width"], self.movie_info["frame_height"]), 
                                      isColor=True)
    
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def write(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.cap.release()
        self.writer.release()
    
    def get_timestamp(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
    
    def get_total_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))