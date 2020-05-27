import os
from tqdm import tqdm
import shutil

counter = 0
for (root, dirs, files) in tqdm(os.walk("D:\DeepFashionDatsets\LowRes\img\WOMEN", topdown=False)):
    for i, file in enumerate(files):
        if "full" in file or "front" in file:

            shutil.copy(root + "\\" + file, f"SimplifiedDatasetDeepFashion/{counter}-df-F.jpg")
            counter += 1
