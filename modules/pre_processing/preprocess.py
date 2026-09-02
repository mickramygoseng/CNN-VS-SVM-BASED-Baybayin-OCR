import os
import pandas as pd
from get_image_paths import Fetch
from preprocess_images import Preprocess

#paths
raw_images_directory = 'data/raw_dataset/Grouped'
save_dataset = 'data/processed_dataset'

#preprocessing
character_metadata = Fetch(raw_images_directory).get_image()
preprocess = Preprocess()

data = {
    "Character" : [],
    "Binary" : []
}

character_group_tracker = 0

os.system('cls')

for character in character_metadata:

    character_group = os.path.basename(character['Character'])
    character_sample_paths = character['ImagePaths']

    character_group_tracker += 1
    pre_processed_number = 0

    for character_sample in character_sample_paths:
        pre_processed_number += 1
        binary_image = preprocess.begin(character_sample)

        data["Character"].append(character_group)
        data["Binary"].append(binary_image)

        print(
            f'\rPhase 2: Group {character_group_tracker}/{len(character_metadata)} ("{character_group}") || '
            f'Sample {pre_processed_number}/{len(character_sample_paths)}        ', 
            end='', 
            flush=True
        )

df = pd.DataFrame(data)

try:
    df.to_pickle(os.path.join(save_dataset, "dataset.pkl"))
    print("[SUCCESS] : dataset.pkl saved")
    df.to_csv(os.path.join(save_dataset, "dataset.csv"))
    print("[SUCCESS] : dataset.csv saved")
except Exception as e:
    print("[FAILED] : dataset.pkl not saved")
    print(f"[DEBUG] : {e}")

