import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class SVM:
    def __init__(self, training: str, test: str) -> None:
        self.train_dataset = pd.read_pickle(training)
        self.test_dataset = pd.read_pickle(test)

        self.model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )

    @staticmethod
    def _prepare(
        df: pd.DataFrame
    ) -> tuple[np.ndarray, pd.Series]:

        
        x = np.stack(
            df["Binary"]
            .apply(lambda img: img.flatten())
            .to_numpy()
        )

        y = df["Character"]

        return x, y

    def train(self) -> None:
        x_train, y_train = self._prepare(self.train_dataset)

        self.model.fit(x_train, y_train)

    def evaluate(self) -> tuple[float, float, float, float]:
        x_test, y_test = self._prepare(self.test_dataset)

        y_pred = self.model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        return accuracy, precision, recall, f1

    def run(self) -> tuple[float, float, float, float]:
        self.train()
        return self.evaluate()