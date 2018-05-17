import cv2
import numpy as np
import classification
from classification import Classifier
from scipy.ndimage.measurements import label, find_objects
from tqdm import tqdm

MIN_WINDOW_SIZE = 50
MAX_WINDOW_SIZE = 250
WINDOW_SCALE_STEP = 50
HORIZON_LINE = 440

class Pipeline:

    def __init__(self):
        self.classifier = classification.load()

    def sliding_windows():
        windowSizes = range(
            MIN_WINDOW_SIZE,
            MAX_WINDOW_SIZE + WINDOW_SCALE_STEP,
            WINDOW_SCALE_STEP)
        for windowSize in windowSizes:
            columnShift = windowSize // 3
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

    def build_heat_map(self, image):
        heatMap = np.zeros((720, 1280), dtype=np.uint8)
        for windowMask in Pipeline.sliding_windows():
            windowImage = image[windowMask]
            windowImage = cv2.resize(windowImage, (64, 64))
            prediction = self.classifier.predict(windowImage)
            if prediction:
                heatMap[windowMask] += 1
        return heatMap

    def process(self, image):
        heatMap = self.build_heat_map(image)
        heatMap[heatMap <= 1] = 0
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

def count_frames(fileName):
    print("Counting frames ...")
    counter = 0
    video = cv2.VideoCapture(fileName)
    while True:
        ret, frame = video.read()
        if not ret: break
        counter += 1
    video.release()
    return counter

if __name__ == "__main__":

    pipeline = Pipeline()

    inputFileName = "project_video.mp4"
    outputFileName = "output.avi"
    frameNumber = count_frames(inputFileName)

    inputVideo = cv2.VideoCapture(inputFileName)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    outputVideo = cv2.VideoWriter(outputFileName, fourcc, 25, (1280, 720))

    for frameId in tqdm(range(frameNumber)):
        ret, image = inputVideo.read()
        image = pipeline.process(image)
        outputVideo.write(image)

    inputVideo.release()
    outputVideo.release()
