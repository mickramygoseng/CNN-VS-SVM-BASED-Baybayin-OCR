import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

class Sample:
    def __init__(self, mainDataset:str, targetDir:str) -> None:
        # essentials
        self.df = pd.read_pickle(mainDataset)

        #paths
        self.main_sampling_folder = targetDir
    
    def RSKF(self, folds:int, repeat:int, seed:int) -> None:
        X = self.df.drop(columns="Character")
        y = self.df["Character"]

        rskf = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeat,
            random_state=seed
        )

        if not os.path.exists(self.main_sampling_folder):
            os.makedirs(self.main_sampling_folder)
        
        for split, (train_idx, test_idx) in enumerate(rskf.split(X, y), start=1):
            #identifying repetition and fold count
            repetition = (split - 1)// folds + 1
            fold = (split - 1) % folds + 1

            print(
                f'\r[DEBUG] Iteration : {repetition} || '
                f'Fold {fold}                                         ', 
                end='', 
                flush=True
            )

            #paths
            repetition_folder = os.path.join(self.main_sampling_folder, f"Iteration {repetition}")
            fold_folder = os.path.join(repetition_folder, f"Fold {fold}")

            #generate dirs
            if not os.path.exists(repetition_folder):
                os.makedirs(repetition_folder)
            
            if not os.path.exists(fold_folder):
                os.makedirs(fold_folder)

            #save folds
            train_df = self.df.iloc[train_idx]
            test_df = self.df.iloc[test_idx]

            try:
                train_df.to_pickle(os.path.join(fold_folder, "train.pkl"))
                test_df.to_pickle(os.path.join(fold_folder, "test.pkl"))


            except Exception as e:
                print(f"[DEBUG] : {e}")



        



        