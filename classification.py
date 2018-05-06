def load_training_data():
    raise NotImplementedError()

def train_classifier():
    raise NotImplementedError()

if __name__ == "__main__":

    trainingData = load_training_data()
    classifier = train_classifier(trainingData)
