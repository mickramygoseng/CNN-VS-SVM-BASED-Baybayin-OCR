import pandas as pd
import glob as glob
import os

svm_dataset = 'data/processed_dataset/svm_dataset.pkl'
cnn_dataset = 'data/processed_dataset/cnn_dataset.pkl'
sampled = 'data/sampled_dataset'

to_validate = int(input(
    "what to validate : \n"
    "1. Main Dataset\n"
    "2. Sampled Dataset\n"
    "3. SVM vs CNN correspondence\n"
    "choice: "
))

if to_validate == 1:
    svm_df = pd.read_pickle(svm_dataset)
    print("svm_dataset:")
    print(svm_df.shape)
    print(svm_df['Character'].value_counts())
    print(svm_df['Features'].iloc[0].shape)

    cnn_df = pd.read_pickle(cnn_dataset)
    print("svm_dataset:")
    print(cnn_df.shape)
    print(cnn_df['Character'].value_counts())
    print(cnn_df['Pixels'].iloc[0].shape)

elif to_validate == 2:
    train_file = "train.pkl"
    test_file = "test.pkl"

    for root, dirs, files in os.walk(sampled):
        for file in files:
            file_path = os.path.join(root, file)

            parts = root.split(os.sep)
            iteration = parts[-2]
            fold = parts[-1]

            df = pd.read_pickle(file_path)

            print(f"\n[DEBUG] {iteration} || {fold} :")

            if "train" in file:
                if (df["Character"].value_counts() == 800).all():
                    print(f"{file} :")
                    print(f"shape: {df.shape}")
                    print("Status: Pass ✔️")
                else:
                    print("Status: Failed ❌")

            elif "test" in file:
                if (df["Character"].value_counts() == 200).all():
                    print(f"{file} :")
                    print(f"shape: {df.shape}")
                    print("Status: Pass ✔️")
                else:
                    print("Status: Failed ❌")

elif to_validate == 3:
    print("\n[DEBUG] Checking source datasets (svm_dataset.pkl vs cnn_dataset.pkl)")

    svm_df = pd.read_pickle(svm_dataset)
    cnn_df = pd.read_pickle(cnn_dataset)

    print(f"SVM rows: {len(svm_df)} | CNN rows: {len(cnn_df)}")

    same_len = len(svm_df) == len(cnn_df)
    same_index = svm_df.index.equals(cnn_df.index) if same_len else False
    same_labels = (
        (svm_df["Character"].reset_index(drop=True)
         == cnn_df["Character"].reset_index(drop=True)).all()
        if same_len else False
    )
    same_dist = svm_df["Character"].value_counts().sort_index().equals(
        cnn_df["Character"].value_counts().sort_index()
    )

    print(f"Same row count: {same_len}")
    print(f"Same index: {same_index}")
    print(f"Same 'Character' label at every position: {same_labels}")
    print(f"Same class distribution (order-independent): {same_dist}")

    if same_len and same_index and same_labels:
        print("Status: Pass ✔️  (source datasets row-correspond)")
    else:
        print("Status: Failed ❌  (source datasets do NOT row-correspond)")

    print("\n[DEBUG] Checking sampled folds (svm/ vs cnn/ index membership)")

    svm_root = os.path.join(sampled, "svm")
    cnn_root = os.path.join(sampled, "cnn")

    if not (os.path.exists(svm_root) and os.path.exists(cnn_root)):
        print("Sampled svm/cnn folders not found -- run sample.py first.")
    else:
        all_ok = True
        for root, dirs, files in os.walk(svm_root):
            for file in files:
                if file not in ("train.pkl", "test.pkl"):
                    continue

                svm_path = os.path.join(root, file)
                cnn_path = svm_path.replace(svm_root, cnn_root, 1)

                parts = root.split(os.sep)
                iteration = parts[-2]
                fold = parts[-1]

                if not os.path.exists(cnn_path):
                    all_ok = False
                    print(f"[MISSING] {iteration} || {fold} || {file}: no matching CNN file")
                    continue

                svm_split = pd.read_pickle(svm_path)
                cnn_split = pd.read_pickle(cnn_path)

                same_members = set(svm_split.index) == set(cnn_split.index)

                print(f"\n[DEBUG] {iteration} || {fold} || {file} :")
                if same_members:
                    print("Status: Pass ✔️")
                else:
                    all_ok = False
                    print(f"svm rows: {len(svm_split)} | cnn rows: {len(cnn_split)}")
                    print("Status: Failed ❌  (index sets differ)")

        print(f"\n[RESULT] All sampled folds match between svm and cnn: {all_ok}")

else:
    print("selected none of the above.")