from PIL import Image
import numpy as np

class Preprocess:
    def begin(self, image_path:str) -> np.array:
        img = Image.open(image_path)
        img.resize((50,50))

        img_greyscale = img.convert("L")
        array = np.array(img_greyscale)

        binary = (array > 127).astype(np.uint8)

        return binary
