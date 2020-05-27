import os
import re
import shutil
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
import time


def ResizeToSquare(path):
    desired_size = 1024

    im = Image.open(path)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # new_im = Image.new("RGB", (desired_size, desired_size))
    # new_im.paste(im, ((desired_size - new_size[0]) // 2,
    #                   (desired_size - new_size[1]) // 2))
    #
    # new_im.show()

    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_im = ImageOps.expand(im, padding, fill= (225, 225, 225))

    #new_im.show()
    return new_im



counter = 0
#dirPath = "D:\PythonProjectsDDrive\ClothesTryOnStage2\\traindata\\train"
dirPath = "D:\\Ny mappe\\Ind_clothing_pieces\\Ind_clothing_pieces"
#dirPath = "D:\DeepFashionDatsets\img_highres"
for (root, dirs, files) in tqdm(os.walk(dirPath, topdown=False)):
    for i, file in enumerate(files):
        if file.endswith(".jpg") or file.endswith(".png"):
        #if file == "target.jpg":#bool(re.search("\d\.jpg", file)) or bool(re.search("\d_target\.jpg", file)):
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
            paddedImage.save(f"Ind_clothing_pieces_squared\\{counter}.jpg", "JPEG")#"clothesDataset256\\{counter}.jpg", "JPEG")
            #shutil.copy(root + "\\" + file, f"SimplifiedDataset/{counter}.jpg")

