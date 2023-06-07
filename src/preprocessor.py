import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class Preprocessor:
    """
    This is the Preprocessor class that is used to perform text processing tasks such as tokenization,
    removal of stop words, and lemmatization.
    """

    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_regex = re.compile(r'[^\w\s]')

    def tokenize(self, text):
        if type(text) is not str:
            text = str(text)
        return word_tokenize(text.lower())

    def remove_special_characters(self, words):
        cleaned_words = []
        for word in words:
            cleaned_word = self.punctuation_regex.sub('', word)
            if cleaned_word:
                cleaned_words.append(cleaned_word)
        return cleaned_words

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def lemmatize(self, words):
        lemmatized_words = []
        for word, pos in nltk.pos_tag(words):
            pos_tag = self.get_wordnet_pos(pos)
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=pos_tag)
            lemmatized_words.append(lemmatized_word)
        return lemmatized_words

    def get_wordnet_pos(self, treebank_pos):
        if treebank_pos.startswith('J'):
            return wordnet.ADJ
        elif treebank_pos.startswith('V'):
            return wordnet.VERB
        elif treebank_pos.startswith('N'):
            return wordnet.NOUN
        elif treebank_pos.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun if no specific POS tag is found
