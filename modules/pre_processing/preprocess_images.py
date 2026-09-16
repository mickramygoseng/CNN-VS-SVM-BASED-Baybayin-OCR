from PIL import Image
import numpy as np

class Preprocess:
    def begin(self, image_path:str, model_type:str) -> np.ndarray:
        img = Image.open(image_path)

        if model_type == "SVM":
            img_greyscale = img.convert("L")
            array = np.array(img_greyscale)
            output = (array > 127).astype(np.uint8)

        elif model_type == "CNN":
            array = np.array(img)
            output = array.astype(np.float32) / 255.0

        return output
