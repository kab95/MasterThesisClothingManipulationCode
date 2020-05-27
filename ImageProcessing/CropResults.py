from PIL import Image
from tqdm import tqdm
import os

def CropImageSeries(image):
    IMAGERESOLUTIONS = image.height #Assumes square images
    totalImages = image.width // IMAGERESOLUTIONS
    indImages = []
    for i in range(totalImages):
        tmpImage = image.copy()
        cropped_img = tmpImage.crop((i*IMAGERESOLUTIONS, 0, (i+1)*IMAGERESOLUTIONS, IMAGERESOLUTIONS))
        #cropped_img.show()
        indImages.append(cropped_img)
    return indImages


counter = 0
dirPath = "MiscImages\ImageSeriesResults"
savePath = "MiscImages\ImageSeriesResultsCroppedIndividuals"
for (root, dirs, files) in tqdm(os.walk(dirPath, topdown=False)):
    for i, file in enumerate(files):
        if file.endswith(".jpg") or file.endswith(".png"):#bool(re.search("\d\.jpg", file)) or bool(re.search("\d_target\.jpg", file)):
            counter += 1

            imgSeries = Image.open(root + "\\" + file)
            individualImages = CropImageSeries(imgSeries)

            for i, indImg in enumerate(individualImages):
                indImg.save(savePath + "\\" + f"series{str(counter).zfill(3)}_{str(i).zfill(3)}.png")


