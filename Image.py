import numpy as np
from PIL import Image

img = Image.open("mol_img/test/0/0/0/000037687605.png")
print(img.mode)

img_array = np.array(img)
print(img_array.shape)
print(img_array.dtype)
print(img_array)

img_id = "000037687605"
print(img_id[0])
print(img_id[1])
print(img_id[-1])
