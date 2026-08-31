import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SVM:
    def __init__(self, training:str, test:str) -> None:
        #datasets
        self.train_dataset = pd.read_pickle(training)
        self.test_dataset = pd.read_pickle(test)

    def run(self):
        #label and value splits then flatten binary vector images from 2d to 1d
        x_train = np.stack(
            self.train_dataset.
            drop(columns="Character").
            iloc[:, 0].apply(
                lambda x: x.flatten()
            )
        )
        y_train = self.train_dataset["Character"]

        x_test = np.stack(
            self.test_dataset.
            drop(columns="Character").
            iloc[:, 0].apply(
                lambda x: x.flatten()
            )
        )
        y_test = self.test_dataset["Character"]

        #initialize and fit svm model
        svm = LinearSVC(
            dual=False, 
            max_iter=2000
        )
        svm.fit(x_train, y_train)

        #prediction
        y_pred = svm.predict(x_test)

        #evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        return accuracy, precision, recall, f1