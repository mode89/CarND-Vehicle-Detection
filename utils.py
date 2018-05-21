import cv2
from tqdm import tqdm

FPS = 25

def count_frames(fileName):
    print("Counting frames ...")
    counter = 0
    video = cv2.VideoCapture(fileName)
    while video.grab():
        counter += 1
    video.release()
    return counter

class VideoClip:

    def __init__(self, fileName, start=0, stop=0):
        totalframeNumber = count_frames(fileName)
        self.frameNumber = min(stop * FPS, totalframeNumber) \
            if stop > 0 else totalframeNumber
        self.video = cv2.VideoCapture(fileName)
        if start > 0:
            self.frameNumber -= start * FPS
            print("Skipping frames ...")
            for frameId in range(start * FPS):
                self.video.grab()

    def __del__(self):
        print("Releasing video ...")
        self.video.release()

    def frames(self):
        for frameId in tqdm(range(self.frameNumber)):
            ret, frame = self.video.read()
            if ret:
                yield frame
            else:
                break
