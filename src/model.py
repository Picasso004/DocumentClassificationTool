import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import classification_report


class Model:
    def __init__(self, dataset=None, classifier=None, model=None):
        if model is None:
            self.dataset = dataset
            self.classifier = classifier
        else:
            self.load_model(model.replace("{", "_").replace("|", "_").replace("}", ""))

    def save_model(self):
        model_name = 'model_'+f"{type(self.dataset.vectorizer).__name__.replace('Vectorizer', '')}_" + \
                     f"{type(self.classifier).__name__}"
        print(f"Exporting model: {model_name}...")

        # Create the 'models' directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models/', exist_ok=True)

        # Save the model
        with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                  + f'/models/{model_name}.pkl', 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, model_name):
        print(f"Loading model {model_name}...")
        filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'/models/{model_name}.pkl'
        with open(filename, 'rb') as file:
            deserialized_obj = pickle.load(file)
        # Assign attributes of deserialized object to current instance
        self.dataset = deserialized_obj.dataset
        self.classifier = deserialized_obj.classifier

    def train_model(self):
        self.dataset.preprocess_data()
        self.dataset.vectorize_data()

        # Split the data into training and testing sets
        print("Spliting data...")
        train_data = np.array(self.dataset.train_data['features'].tolist())
        train_target = np.array(self.dataset.train_data['topic'])
        test_data = np.array(self.dataset.test_data['features'].tolist())
        test_target = np.array(self.dataset.test_data['topic'])

        # Train the models
        print(f"Training model using {type(self.classifier).__name__}...")
        self.classifier.train(train_data, train_target)

        # Predict labels for the test data
        predictions = self.classifier.predict(test_data)

        # Evaluate the models's performance
        accuracy = self.classifier.evaluate(test_target, predictions)
        report = classification_report(test_target, predictions, zero_division=1)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

    def predict(self, text):
        data = pd.DataFrame([{'id': None, 'topic': None, 'text': text, 'length': len(text)}])
        data['text'].apply(self.dataset.preprocess_input)
        vectors = self.dataset.vectorize_input(data)
        if type(vectors) is not list:
            vectors = vectors.tolist()
        data = np.array(vectors)
        prediction = self.classifier.predict(data)[0]
        print("All done")
        print(f"Result : {prediction}")
        return prediction


