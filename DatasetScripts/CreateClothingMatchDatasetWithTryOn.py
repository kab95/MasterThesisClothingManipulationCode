import os
import re
import shutil
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
import time
from mtcnn import MTCNN
import cv2

def ResizeToSquare(path):
    desired_size = 256

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

def CropImageTop(topCoordinate, imCrop):
    width, height = imCrop.size

    # Cropped image of above dimension
    # (It will not change orginal image)
    im1 = imCrop.crop((0, topCoordinate, width, height))

    # Shows the image in image viewer
    return im1

def CutFromBackgroundRatio(path):
    BACKGROUNDVALUE = 630
    NECKPERCENTDECREASE = 5

    im = Image.open(path)
    width, height = im.size
    imagePixels = im.getdata()

    bCounter, mostB, neckLine = 0, 0, 0
    for i, pix in enumerate(imagePixels):
        if pix[0]+pix[1]+pix[2] >= BACKGROUNDVALUE: bCounter += 1
        #print(pix)

        if i % width == 0:
            personPercent = round(100 - (bCounter / width) * 100)
            #print(f"{personPercent}%")

            if mostB < personPercent and personPercent != 99: mostB = personPercent
            if personPercent < mostB - NECKPERCENTDECREASE:
                neckLine = i // width
                break
            bCounter = 0
    #im.show()
    croppedImage = CropImageTop(neckLine, im)
    #croppedImage.show()
    #print(neckLine)
    #print("Image Finished.")
    return croppedImage

def CutFromDetectFace(path, detector):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    face = detector.detect_faces(img)
    if len(face) == 0:
        print("Failed facial recognition, trying manual algorithm...")
        return None

    neckLine = face[0]['box'][0] - 15 if face[0]['box'][0] - 15 > 0 else 0
    croppedImage = CropImageTop(neckLine, Image.open(path))
    #croppedImage.show()
    return croppedImage


counter = 0
for (root, dirs, files) in tqdm(os.walk("D:\PythonProjectsDDrive\ClothesTryOnStage2\\traindata\\train", topdown=False)):
    for i, file in enumerate(files):
        if bool(re.search("\d\.jpg", file)):
            counter += 1

            #detector = MTCNN()
            #outImage = CutFromDetectFace(root + "\\" + file, detector)
            if outImage == None:
                outImage = CutFromBackgroundRatio(root + "\\" + file)
            #paddedImage = ResizeToSquare(root + "\\" + file)

            #paddedImage.thumbnail((128, 128), Image.ANTIALIAS)
            #paddedImage.save(f"SimplifiedDatasetResized\\{counter}.jpg", "JPEG")
            outImage.save(f"facelessDataset\\{counter}.jpg", "JPEG")
            outImage.close()
