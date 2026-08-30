import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import models, layers  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CNN:
    def __init__(self, train: str, test: str, encoder: LabelEncoder) -> None:
        self.train_dataset = pd.read_pickle(train)
        self.test_dataset = pd.read_pickle(test)
        self.encoder = encoder 

    def run(self):
        # label and value splits
        x_train = np.stack(self.train_dataset["Binary"].to_numpy())
        x_test = np.stack(self.test_dataset["Binary"].to_numpy())

        x_train = x_train.reshape(-1, 32, 32, 1).astype("float32") 
        x_test = x_test.reshape(-1, 32, 32, 1).astype("float32")

        y_train = self.encoder.transform(self.train_dataset["Character"]) 
        y_test = self.encoder.transform(self.test_dataset["Character"])
        num_classes = len(self.encoder.classes_)

        # set up CNN model
        cnn = models.Sequential([
            layers.Input(shape=(32, 32, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes)
        ])

        cnn.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        # model fit + prediction
        cnn.fit(x_train, y_train, epochs=10, validation_split=0.1)
        y_pred = np.argmax(cnn.predict(x_test), axis=1)

        # evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        return accuracy, precision, recall, f1