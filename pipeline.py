import cv2
import numpy as np
import classification
from classification import Classifier
from scipy.ndimage.measurements import label, find_objects
from tqdm import tqdm
import utils

WINDOW_SIZES = [ 30, 40, 60, 90, 120, 180 ]
HORIZON_LINE = 440

class Pipeline:

    def __init__(self):
        self.classifier = classification.load()
        self.heatMap = np.zeros((720, 1280), dtype=np.float32)

    def sliding_windows():
        for windowSize in WINDOW_SIZES:
            columnShift = windowSize // 4
            columnNum = (1280 - windowSize) // columnShift + 1
            rowShift = windowSize // 4
            for column in range(columnNum):
                for row in range(-1, 2):
                    top = HORIZON_LINE - windowSize // 3 - row * rowShift
                    bottom = top + windowSize
                    left = column * columnShift
                    right = left + windowSize
                    windowMask = np.ix_(
                        range(top, bottom),
                        range(left, right))
                    yield windowMask

    def update_heat_map(self, image):
        self.heatMap *= 0.9
        for windowMask in Pipeline.sliding_windows():
            windowImage = image[windowMask]
            windowImage = cv2.resize(windowImage, (64, 64))
            prediction = self.classifier.predict(windowImage)
            if prediction > 0.1:
                self.heatMap[windowMask] += (0.01 * prediction)
        self.heatMap = np.clip(self.heatMap, 0.0, 1.0)

    def process(self, image):
        self.update_heat_map(image)

        heatMap = np.uint8(self.heatMap * 255.0)
        heatMap[heatMap < 25] = 0
        labelMap, labels = label(heatMap)

        for labeledArea in Pipeline.find_objects(labelMap):
            boundingBox = ObjectBoundingBox(labeledArea)
            cv2.rectangle(image,
                pt1=(boundingBox.left, boundingBox.top),
                pt2=(boundingBox.right, boundingBox.bottom),
                color=(255, 0, 0),
                thickness=3)

        return image

    def find_objects(labelMap):
        objects = find_objects(labelMap)
        objects = Pipeline.filter_tall_objects(objects)
        return objects

    def filter_tall_objects(objects):
        for obj in objects:
            boundingBox = ObjectBoundingBox(obj)
            width = boundingBox.right - boundingBox.left
            height = boundingBox.bottom - boundingBox.top
            if width / height > 0.8:
                yield obj

class ObjectBoundingBox:

    def __init__(self, obj):
        self.top = obj[0].start
        self.bottom = obj[0].stop
        self.left = obj[1].start
        self.right = obj[1].stop

if __name__ == "__main__":

    pipeline = Pipeline()

    inputFileName = "project_video.mp4"
    outputFileName = "output.avi"

    inputVideo = utils.VideoClip(inputFileName)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    outputVideo = cv2.VideoWriter(outputFileName, fourcc, 25, (1280, 720))

    for frame in inputVideo.frames():
        image = pipeline.process(frame)
        outputVideo.write(image)

    outputVideo.release()
