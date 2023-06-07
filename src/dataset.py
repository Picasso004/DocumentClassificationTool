import os
import pandas as pd
from src.preprocessor import Preprocessor
from bs4 import BeautifulSoup


class Dataset:
    def __init__(self, vectorizer=None):
        self.train_data = []
        self.test_data = []
        self.preprocessor = Preprocessor()
        self.vectorizer = vectorizer

    def load_data(self, path):
        print(f"Loading data from {path} ...")
        if not os.path.exists(path):
            print(f"{path} doesn't exist")
            return
        for filename in os.listdir(path):
            if filename.endswith('.sgm'):
                print(f"Reading file{filename}...")
                try:
                    with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    print(f"Error : 'utf-8' codec can't decode. Skipping file{filename}...")

                    continue

                soup = BeautifulSoup(content, 'html.parser')
                reuters = soup.findAll('reuters')

                for reuter in reuters:
                    if reuter['topics'] == "YES" and reuter.topics.text != '' and reuter.body is not None:
                        topic = reuter.topics.d.text
                        lewissplit = reuter['lewissplit']
                        if lewissplit.casefold() == 'train':
                            self.train_data.append({'id': reuter['newid'], 'topic': topic, 'text': reuter.body.text,
                                                    'length': len(reuter.body.text)})
                        elif lewissplit.casefold() == 'test':
                            self.test_data.append({'id': reuter['newid'], 'topic': topic, 'text': reuter.body.text,
                                                   'length': len(reuter.body.text)})

    def preprocess_data(self):
        print("Preprocessing data...")
        for data in [self.train_data, self.test_data]:
            for doc in data:
                words = self.preprocessor.tokenize(doc['text'])
                words = self.preprocessor.remove_stopwords(words)
                words = self.preprocessor.remove_special_characters(words)
                words = self.preprocessor.lemmatize(words)
                doc['text'] = words

        self.train_data = pd.DataFrame(self.train_data)
        self.test_data = pd.DataFrame(self.test_data)

    def vectorize_data(self):
        print(f"Vectorizing data using {type(self.vectorizer).__name__}...")
        print("Fiting data...")
        self.vectorizer.fit(self.train_data)
        print("Vectorizing training data...")
        self.train_data['features'] = self.vectorizer.transform(self.train_data)
        print("Vectorizing test data...")
        self.test_data['features'] = self.vectorizer.transform(self.test_data)

    def preprocess_input(self, text):
        print("Preprocessing input data...")
        words = self.preprocessor.tokenize(text)
        words = self.preprocessor.remove_stopwords(words)
        words = self.preprocessor.remove_special_characters(words)
        words = self.preprocessor.lemmatize(words)
        return words

    def vectorize_input(self, dataframe):
        print(f"Vectorizing data using {type(self.vectorizer).__name__}...")
        print("Vectorizing input data...")
        return self.vectorizer.transform(dataframe)
