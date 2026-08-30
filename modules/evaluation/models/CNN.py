import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import models, datasets, layers

class CNN:
    def __init__(self, train:str, test:str) -> None:
        self.train_dataset = pd.read_pickle(train)
        self.test_dataset = pd.read_pickle(test)
    
    def run(self):
        #label and value splits
        x_train = self.train_dataset.drop(columns="Character")
        y_train = self.train_dataset["Character"]

        x_test = self.test_dataset.drop(columns="Character")
        y_test = self.test_dataset["Character"]

        #set up CNN model
        cnn = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(59)
        ])

        cnn.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
        
