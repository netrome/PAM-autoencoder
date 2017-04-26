import PIL
import glob
import sys
import os

# Get globals
CENTER_BLACK = "center" in sys.argv

# define paths
place = os.getcwd()
lfw_path = place + "/lfw-deepfunneled/"
pattern_path = place + "/patterns/"
target_path = place + "/targets/"

# Clear directories
os.system("rm targets/*")
os.system("rm patterns/*")

k = 0
n = 100
for path, dirs, files in os.walk(lfw_path):
    if len(files) > 0 and k < n:
        for i in files:
            image_path = path + "/" + i

            # Save cropped original image
            os.system("convert -resize 64x64 {0} /tmp/img.jpg".format(image_path))

            os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
            os.system("convert /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
            k += 1

            if CENTER_BLACK:
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert -stroke black -draw \"rectangle 24, 24, 40, 40\" /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
                k += 1
                

