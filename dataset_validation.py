import pandas as pd
import glob as glob
import os

main_dataset = 'data/processed_dataset/dataset.pkl'
sampled = 'data/sampled_dataset'

to_validate = int(input("what to validate : \n1. Main Dataset\n2. Sampled Dataset\nchoice: "))

if to_validate == 1:
    df = pd.read_pickle(main_dataset)

    print(df.shape)
    print(df['Character'].value_counts())
    print(df['Binary'].iloc[0].shape)

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

else:
    print("selected none of the above.")