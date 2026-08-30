import glob
import os

class Fetch:
    def __init__(self, path:str):
        self.directory_path = path

    def get_image(self) -> list[dict]:
        image_dirs = glob.glob(os.path.join(self.directory_path, '*'))
        image_paths = []

        for image_dir in image_dirs:
            print(
                '\r Phase 1: Data Fetching || '
                f'Fetching Baybayin Character Samples: {os.path.basename(image_dir)}',
                end='', 
                flush=True
            )

            image_path = glob.glob(os.path.join(image_dir, '**.jpg'))
            data = {
                "Character": os.path.basename(image_dir),
                "ImagePaths": image_path,
            }
            image_paths.append(data)
            
        return image_paths