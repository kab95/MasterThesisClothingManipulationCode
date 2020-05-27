from PIL import Image
import os

TOTALIMAGESWIDTH = 10
TOTALIMAGESHEIGHT = 5

position_counter = 0
amlgImg = None
#dirPath = "D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\DatasetProcesssing\Combined256Dataset"
#dirPath = "D:\DeepFashion1024v2Curated6K\DeepFashion1024v2"
#dirPath = "D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\ImageEvaluation\\tmpNegative"
dirPath = "D:\\FjongImages\\Clothes_Fjong_Squared_1024"
for subdir, dirs, files in os.walk(dirPath):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            print(f"pasting image {position_counter} out of {TOTALIMAGESWIDTH * TOTALIMAGESHEIGHT}")

            im = Image.open(filepath)
            if position_counter == 0: amlgImg = Image.new('RGB', (im.width * TOTALIMAGESWIDTH, im.height * TOTALIMAGESHEIGHT))

            amlgImg.paste(im, (im.width * (position_counter % TOTALIMAGESWIDTH), im.height * (position_counter // TOTALIMAGESWIDTH)))
            position_counter += 1
        if position_counter >= TOTALIMAGESWIDTH * TOTALIMAGESHEIGHT: break
amlgImg.thumbnail((2048, 1024))
directory = dirPath.split("\\")[-1]
amlgImg.save(f"D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\Collages\\{directory}_collage_{TOTALIMAGESHEIGHT}by{TOTALIMAGESWIDTH}.jpg", "JPEG")
amlgImg.show()

