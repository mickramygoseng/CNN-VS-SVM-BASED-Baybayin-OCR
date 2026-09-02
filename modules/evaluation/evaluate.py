from models.CNN import CNN
from models.SVM import SVM
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# variables
sampled_dataset_paths = 'data/sampled_dataset'
save_path = "data/metrics_dataset"
metrics_dataset = os.path.join(save_path, "model_metrics.csv")
models = [ "cnn", "svm" ]
n_iterations = 10
n_folds = 5

os.makedirs(save_path, exist_ok=True)

# load existing progress (if any)
if os.path.exists(metrics_dataset):
    df_existing = pd.read_csv(metrics_dataset)
    all_data = df_existing.to_dict(orient="list")
    completed = set(
        zip(all_data["model"], all_data["iteration"], all_data["fold"])
    )
else:
    all_data = {
        "model": [],
        "iteration": [],
        "fold": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    completed = set()

def save_progress(accuracy, precision, recall, f1, iteration, fold, model):
    all_data["model"].append(model)
    all_data["iteration"].append(iteration)
    all_data["fold"].append(fold)
    all_data["accuracy"].append(accuracy)
    all_data["precision"].append(precision)
    all_data["recall"].append(recall)
    all_data["f1"].append(f1)

    df = pd.DataFrame(all_data)
    df.to_csv(metrics_dataset, index=False)
    completed.add((model, iteration, fold))

# Create and fit label encoder
encoder = LabelEncoder()

all_labels = []

for iteration in range(1, n_iterations + 1):
    for fold in range(1, n_folds + 1):
        train_path = os.path.join(
            sampled_dataset_paths,
            f"Iteration {iteration}",
            f"Fold {fold}",
            "train.pkl"
        )
        test_path = os.path.join(
            sampled_dataset_paths,
            f"Iteration {iteration}",
            f"Fold {fold}",
            "test.pkl"
        )

        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        all_labels.extend(train_df["Character"].tolist())
        all_labels.extend(test_df["Character"].tolist())

encoder.fit(all_labels)

for model in models:
    for iteration in range(n_iterations):
        for fold in range(n_folds):
            iter_num = iteration + 1
            fold_num = fold + 1

            if (model, iter_num, fold_num) in completed:
                continue

            train_data_path = os.path.join(
                sampled_dataset_paths, f"Iteration {iter_num}", f"Fold {fold_num}", "train.pkl"
            )
            test_data_path = os.path.join(
                sampled_dataset_paths, f"Iteration {iter_num}", f"Fold {fold_num}", "test.pkl"
            )

            print(f"\r\n[DEBUG] Currently: {model} iter {iter_num} fold {fold_num}        ", end='', flush=True)

            if model == "svm":
                runner = SVM(train_data_path, test_data_path)
            else:
               runner = CNN(train_data_path, test_data_path, encoder)

            accuracy, precision, recall, f1 = runner.run()

            save_progress(accuracy, precision, recall, f1, iter_num, fold_num, model)
            print(f"\r\n[DEBUG] Done: {model} iter {iter_num} fold {fold_num}        ", end='', flush=True)