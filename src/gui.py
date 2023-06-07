import sys
import tkinter as tk
import os
from tkinter import ttk
from src.model import Model
from src.dataset import Dataset
import src.vectorizer as vectorizer  # Import the vectorizer module that contains the vectorizer classes
import src.classifier as classifier  # Import the classifier module that contains the classifier classes

# Create a dictionary that maps the selected item string to the corresponding Vectorizer and Classifier classes
model_mapping = {
    "model{BoW|NaiveBayes}": (vectorizer.BoWVectorizer, classifier.NaiveBayes),
    "model{TfIdf|NaiveBayes}": (vectorizer.TfIdfVectorizer, classifier.NaiveBayes),
    "model{BoW|MinDistance}": (vectorizer.BoWVectorizer, classifier.MinDistance),
    "model{TfIdf|MinDistance}": (vectorizer.TfIdfVectorizer, classifier.MinDistance),
    "model{Word2Vec|MinDistance}": (vectorizer.Word2VecVectorizer, classifier.MinDistance),
    "model{BoW|RandomForest}": (vectorizer.BoWVectorizer, classifier.RandomForest),
    "model{TfIdf|RandomForest}": (vectorizer.TfIdfVectorizer, classifier.RandomForest),
    "model{Word2Vec|RandomForest}": (vectorizer.Word2VecVectorizer, classifier.RandomForest),
}


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Classification Tool")

        # Create a list of items for the selector
        items = list(model_mapping.keys())

        # Create a label for the selector
        selector_label = ttk.Label(self.root, text="Select a model:")
        selector_label.pack()

        # Create a selector
        self.selector = ttk.Combobox(self.root, values=items, width=30)
        self.selector.pack()

        # Create a label for the text zone
        text_label = ttk.Label(self.root, text="Enter or paste text:")
        text_label.pack()

        # Create a text zone
        self.text_zone = tk.Text(self.root, height=10, width=50)
        self.text_zone.pack()

        # Create a label for the result
        self.result_label = ttk.Label(self.root, text="Result:")
        self.result_label.pack()

        # Create a label for the classification
        self.classification_label = ttk.Label(self.root, text="", foreground="green")
        self.classification_label.pack()

        # Create a run button
        run_button = ttk.Button(self.root, text="Run", command=self.run_classification)
        run_button.pack()

    def run_classification(self):
        print("------")
        # Retrieve the selected item and entered/pasted text
        selected_item = self.selector.get()
        entered_text = self.text_zone.get("1.0", tk.END)

        if not selected_item or entered_text.isspace():
            tk.messagebox.showwarning("Document Classification Tool",
                                      "Warning: Please select a model and enter some text")
            return

        # Perform document classification using the selected item and entered/pasted text
        # Just printing the selected item and entered text for demonstration purposes
        print("Selected model:", selected_item)
        print("Entered Text:")
        print(entered_text)

        if not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models/' +
                              selected_item.replace("{", "_").replace("|", "_").replace("}", "") + '.pkl'):
            print(f"{selected_item} is not trained yet")
            print("Staring training")

            # Get the corresponding Vectorizer and Classifier classes from the model_mapping
            vectorizer_class, classifier_class = model_mapping[selected_item]

            # Create instances of the Vectorizer and Classifier
            vectorizer = vectorizer_class()
            classifier = classifier_class()

            # Load and preprocess the data
            dataset = Dataset(vectorizer)

            dataset.load_data(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/reuters21578')

            m = Model(dataset, classifier)
            m.train_model()
            print("Model trained!")
            m.save_model()

        # Get the model
        try:
            model = Model(model=selected_item)
        except MemoryError:
            print("Error : Memory error")
            print()
            tk.messagebox.showerror("Document Classification Tool",
                                      "Error: Memory error"
                                      "Please try again")
            sys.exit()

        result = model.predict(entered_text)

        # Set the classification result label
        self.classification_label.configure(text=result)

    def run(self):
        self.root.mainloop()