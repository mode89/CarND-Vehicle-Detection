import cv2
import numpy as np
import classification
from classification import Classifier
from scipy.ndimage.measurements import label, find_objects
from tqdm import tqdm
import training_data

MIN_WINDOW_SIZE = 50
MAX_WINDOW_SIZE = 250
WINDOW_SCALE_STEP = 50
HORIZON_LINE = 440
WINDOW_HORIZON_SECTION = 1 / 3
WINDOWS_ROWS_NUMBER = 5
WINDOWS_ROWS = range(
    WINDOWS_ROWS_NUMBER // 2,
    WINDOWS_ROWS_NUMBER // 2 + 1)
WINDOW_HORIZONTAL_SHIFT = 1 / 4
WINDOW_VERTICAL_SHIFT = 1 / 4

class Pipeline:

    def __init__(self):
        self.classifier = classification.load()

    def extract_windows_features(image, windowSize):
        imageWidth = image.shape[1]
        horizonShift = int(windowSize * WINDOW_HORIZON_SECTION)
        verticalShift = int(windowSize * WINDOW_VERTICAL_SHIFT)
        horizontalShift = int(windowSize * WINDOW_HORIZONTAL_SHIFT)
        halfRowsNum = WINDOWS_ROWS_NUMBER // 2
        columnsNum = (imageWidth - windowSize) // horizontalShift + 1
        top = HORIZON_LINE - horizonShift - verticalShift * halfRowsNum
        bottom = top + windowSize + verticalShift * halfRowsNum * 2
        right = windowSize + (columnsNum - 1) * horizontalShift
        roiImage = image[top:bottom,:right,:]
        roiScaledSize = (
            int((1 + (columnsNum - 1) * WINDOW_HORIZONTAL_SHIFT) * 64),
            int((1 + 2 * halfRowsNum * WINDOW_VERTICAL_SHIFT) * 64)
        )
        roiImage = cv2.resize(roiImage, dsize=roiScaledSize)
        roiFeatures = training_data.obtain_features(roiImage)

    def sliding_windows(image):
        windowSizes = range(
            MIN_WINDOW_SIZE,
            MAX_WINDOW_SIZE + WINDOW_SCALE_STEP,
            WINDOW_SCALE_STEP)
        for windowSize in windowSizes:
            roiImage = Pipeline.extract_windows_features(image, windowSize)
            columnShift = windowSize // 4
            columnNum = (1280 - windowSize) // columnShift
            rowShift = windowSize // 5
            for column in range(columnNum):
                for row in range(-2, 3):
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
            prediction = self.classifier.predict(windowImage)
            if prediction:
                heatMap[windowMask] += 1
        return heatMap

    def process(self, image):
        heatMap = self.build_heat_map(image)
        heatMap[heatMap <= 10] = 0
        labelMap, labels = label(heatMap)

        for labeledArea in find_objects(labelMap):
            top = labeledArea[0].start
            bottom = labeledArea[0].stop
            left = labeledArea[1].start
            right = labeledArea[1].stop
            cv2.rectangle(image,
                pt1=(left, top),
                pt2=(right, bottom),
                color=(255, 0, 0),
                thickness=3)

        return image

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

    fileName = "test_video.mp4"
    frameNumber = count_frames(fileName)

    inputVideo = cv2.VideoCapture(fileName)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    outputVideo = cv2.VideoWriter("output.avi", fourcc, 25, (1280, 720))

    for frameId in tqdm(range(frameNumber)):
        ret, image = inputVideo.read()
        image = pipeline.process(image)
        outputVideo.write(image)

    inputVideo.release()
    outputVideo.release()
