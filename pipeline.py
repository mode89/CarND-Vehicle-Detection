import cv2
import numpy as np
import classification
from classification import Classifier
from scipy.ndimage.measurements import label, find_objects

MIN_WINDOW_SIZE = 50
MAX_WINDOW_SIZE = 250
WINDOW_SCALE_STEP = 50
HORIZON_LINE = 440
WINDOW_HORIZON_RELATIVE_SHIFT = 0.33

class Pipeline:

    def __init__(self):
        self.classifier = classification.load()

    def process(self, image):
        heatMap = np.zeros((720, 1280), dtype=np.uint8)

        windowSizes = range(
            MIN_WINDOW_SIZE,
            MAX_WINDOW_SIZE + WINDOW_SCALE_STEP,
            WINDOW_SCALE_STEP)
        for windowSize in windowSizes:
            for x in range(0, 1280 - windowSize, windowSize // 2):
                top = HORIZON_LINE - \
                    int(WINDOW_HORIZON_RELATIVE_SHIFT * windowSize)
                bottom = HORIZON_LINE + \
                    int((1 - WINDOW_HORIZON_RELATIVE_SHIFT) * windowSize)
                left = x
                right = x + windowSize
                index = np.ix_(range(top, bottom), range(left, right))
                im = image[index]
                prediction = self.classifier.predict(im)
                if prediction:
                    heatMap[index] += 1

        heatMap[heatMap <= 1] = 0
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

if __name__ == "__main__":

    pipeline = Pipeline()

    image = cv2.imread("test_images/test1.jpg")
    image = pipeline.process(image)

    cv2.imshow("image", image)
    cv2.waitKey(0)
