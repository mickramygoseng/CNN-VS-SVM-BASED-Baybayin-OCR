# SVM vs CNN Based Baybayin OCR

This repository contains the python pipeline used in our undergraduate thesis titled <u>Comparative analysis between Neural Network and Classical Machine Learning in Developing Baybayin Optical Character Recognition Systems</u>

## 🖥️ Installation

**STEP 1: Clone this Repository**

```
git clone git@github.com:mickramygoseng/CNN-VS-SVM-BASED-Baybayin-OCR.git  
```

**STEP 2: Install Dependencies**
```
pip install -r requirements.txt
```

**NOTE:** it is highly suggested to use Python 3.13.15 to ensure maximum compatibility of all dependencies

## 🧩 Running the Pipeline

**STEP 1: Install Baybayin Character Samples**

this study uses the novelty dataset from (Vilvar et al., 2022), you can download it via kaggle using this [link](https://www.kaggle.com/datasets/danielhammond/baybayin-character-dataset)

extract only the grouped dataset and make sure to extract it in **data\raw_dataset**

**STEP 2: Convert the Image files into Binary Vector**

```
python modules\pre_processing\preprocess.py
```

**STEP 3: Apply Repeated Stratified K-fold Cross Validation**

```
python modules\sampling\sample.py
```

**STEP 4: Train and Evaluate SVM and CNN models**
```
python modules\evaluation\evaluate.py
```

## 🔎 Validating Dataset Sanity
To validate the dataset sanity of the pre-processed dataset as well as the sampled dataset applied with repeated k-fold technique, use the following bash script

```
python dataset_validation.py
```

## 📗 About the Dataset

this study makes use of the novelty dataset from (Vilvar et al., 2022) to train and validate its proposed Baybayin OCR systems, the dataset is composed of 59,000 baybayin character samples (1000 per character), in this study, the dataset will be utilized as folllows:
- Training - 80%
- Test - 20%

to analyze the frameworks developed in this study across multiple performance meterics, the paper utilizes repeated stratified k-fold cross validation, with 5 folds repeated 10 times

## 📃 About the Study

**Statistical Tools Applied**
- Support Vector Machine
- Convolutional Neural Network
- MANOVA

**Objectives**
- To Develop Baybayin Optical Character Recognition frameworks using Classical Machine Learning and Neural Network models
- To Characterize the performance between the frameworks across multiple performance metrics
- To Determine if there is a significant difference between the performance of the proposed frameworks
- To Determine which performance metrics significantly differ between the proposed frameworks

## 👥 About Us

We are a group of 4th year BS Applied Mathematics students from Cavite State University - Don Severino Delas Alas Campus

This repository was developed not only to support our undergraduate thesis efforts but also to contribute a reusable and research-oriented pipeline for future researchers interested in expanding works related to baybaying OCR systems.