import PIL
import glob
import sys
import os

# Get globals
CENTER_BLACK = "center" in sys.argv
NORMAL = "normal" in sys.argv

# define paths
place = os.getcwd()
lfw_path = place + "/lfw-deepfunneled/"
pattern_path = place + "/patterns_highres/"
target_path = place + "/targets_highres/"

# Clear directories
os.system("rm targets_highres/*")
os.system("rm patterns_highres/*")

k = 0
n = 1000
for path, dirs, files in os.walk(lfw_path):
    if len(files) > 0 and k < n:
        for i in files:
            image_path = path + "/" + i

            # Save cropped original image
            os.system("convert -resize 120x120 {0} /tmp/img.jpg".format(image_path))

            if NORMAL:
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
                k += 1

            if CENTER_BLACK:
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert -stroke black -draw \"rectangle 40, 40, 80, 80\" /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
                k += 1
    elif k == n:
        break
                

