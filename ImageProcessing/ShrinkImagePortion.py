import os
import re
import shutil
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
import time


def ResizeToSquare(path):
    newWidth = 1536
    newHeight = 1536

    shrinkImg = Image.open(path)
    shrinkImg.thumbnail((768, 1024), Image.ANTIALIAS)
    emptyImg = Image.new("RGB", (newWidth, newHeight))
    emptyImg.paste(shrinkImg.getpixel((1,1)), [0,0,newWidth,newHeight])
    emptyImg.paste(shrinkImg, (int(newWidth * 0.85) // 3, 200))
    emptyImg.show()
    return emptyImg



counter = 0
dirPath = "D:\FjongImages\Items on hanger"
#dirPath = "D:\DeepFashionDatsets\img_highres"
for (root, dirs, files) in tqdm(os.walk(dirPath, topdown=False)):
    for i, file in enumerate(files):
        if file.endswith(".jpg"):#bool(re.search("\d\.jpg", file)) or bool(re.search("\d_target\.jpg", file)):
            counter += 1
            #print(root + dirs[0] + "\\" + file)
            #resChangeImg = Image.open(root + "\\" + file)
            paddedImage = ResizeToSquare(root + "\\" + file)
            #print(resChangeImg.size)
            #if resChangeImg.size != (192, 256):
            #    print(resChangeImg.size, counter)


            paddedImage.thumbnail((1024, 1024), Image.ANTIALIAS)
            #print(resChangeImg.size)
            #print(paddedImage.size)
            paddedImage.save(f"DatasetProcesssing\ShrunkClothing\\{counter}.jpg", "JPEG")
            #shutil.copy(root + "\\" + file, f"SimplifiedDataset/{counter}.jpg")

