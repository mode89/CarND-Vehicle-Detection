import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import training_data

CLASSIFIER_FILE_NAME = "classifier.pkl"

class Classifier:

    def train(self, trainingData):
        print("Splitting training data ...")
        trainX, testX, trainY, testY = train_test_split(
            trainingData[0], trainingData[1],
            test_size=0.2, random_state=42)

        print("Scaling training data ...")
        self.scaler = StandardScaler().fit(trainX)
        trainX = self.scaler.transform(trainX)
        testX = self.scaler.transform(testX)

        print("Training classifier ...")
        self.svc = LinearSVC()
        self.svc.fit(trainX, trainY)

        trainScore = self.svc.score(trainX, trainY)
        testScore = self.svc.score(testX, testY)
        print("Training score: {:.3f}".format(trainScore))
        print("Test score: {:.3f}".format(testScore))

    def predict(self, image):
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = training_data.obtain_features(image)
        features = self.scaler.transform([features])
        return self.svc.predict(features)

def save(classifier):
    print("Saving classifier ...")
    with open(CLASSIFIER_FILE_NAME, "wb") as f:
        pickle.dump(classifier, f)

def load():
    print("Loading classifier ...")
    with open(CLASSIFIER_FILE_NAME, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":

    trainingData = training_data.load()
    classifier = Classifier()
    classifier.train(trainingData)
    save(classifier)
