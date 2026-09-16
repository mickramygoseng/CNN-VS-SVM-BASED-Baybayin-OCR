import os
import pandas as pd
from get_image_paths import Fetch
from preprocess_images import Preprocess

#paths
raw_images_directory = 'data/raw_dataset'
save_dataset = 'data/processed_dataset'

# Essential set up
preprocess = Preprocess()
svm_data = {
    "Character" : [],
    "Features" : []
}
cnn_data = {
    "Character" : [],
    "Pixels" : []
}

# Fetch images
character_metadata = Fetch(raw_images_directory).get_image()

# Begin initial preprocessing
os.system('cls')
character_group_tracker = 0
for character in character_metadata:

    character_group = os.path.basename(character['Character'])
    character_sample_paths = character['ImagePaths']

    character_group_tracker += 1
    pre_processed_number = 0

    for character_sample in character_sample_paths:
        pre_processed_number += 1

        # Preprocess data for SVM models
        svm_preprocessed = preprocess.begin(character_sample, model_type="SVM")
        svm_data["Character"].append(character_group)
        svm_data["Features"].append(svm_preprocessed)

        # Preprocess data for SVM models
        cnn_preprocessed = preprocess.begin(character_sample, model_type="CNN")
        cnn_data["Character"].append(character_group)
        cnn_data["Pixels"].append(cnn_preprocessed)

        print(
            f'\rPhase 2: Group {character_group_tracker}/{len(character_metadata)} ("{character_group}") || '
            f'Sample {pre_processed_number}/{len(character_sample_paths)}        ', 
            end='', 
            flush=True
        )

# Save preprocessed dataset
svm_df = pd.DataFrame(svm_data)
cnn_df = pd.DataFrame(cnn_data)

try:
    svm_df.to_pickle(os.path.join(save_dataset, "svm_dataset.pkl"))
    cnn_df.to_pickle(os.path.join(save_dataset, "cnn_dataset.pkl"))
    print("[SUCCESS] : datasets saved")

except Exception as e:
    print("[FAILED] : datasets not saved")
    print(f"[DEBUG] : {e}")

