import PIL
import glob
import sys
import os
import random

# Get globals
CENTER_BLACK = "center" in sys.argv
RANDOM_BLACK = "random-black" in sys.argv
NORMAL = "normal" in sys.argv
res = int(sys.argv[1]) # resolution

# define paths
place = os.getcwd()
lfw_path = place + "/lfw-deepfunneled/"
pattern_path = place + "/patterns/"
target_path = place + "/targets/"

# Clear directories
os.system("rm targets/*")
os.system("rm patterns/*")

k = 0
n = 1000
for path, dirs, files in os.walk(lfw_path):
    if len(files) > 0 and k < n:
        for i in files:
            image_path = path + "/" + i

            # Save cropped original image
            os.system("convert -resize {1}x{1} {0} /tmp/img.jpg".format(image_path, res))

            if NORMAL:
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
                k += 1

            if CENTER_BLACK:
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert -stroke black -draw \"rectangle 24, 24, 40, 40\" /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k))
                k += 1

            if RANDOM_BLACK:
                x1 = random.randint(int(res/4), int(res/2) - 1)
                y1 = random.randint(int(res/4), int(res/2) - 1)
                x2 = random.randint(int(res*2/4 + 1), int(res*3/4))
                y2 = random.randint(int(res*2/4 + 1), int(res*3/4))
                rect = ", ".join([str(jj) for jj in [x1, y1, x2, y2]])
                os.system("convert /tmp/img.jpg {0}target{1}.jpg".format(target_path, k))
                os.system("convert -stroke black -draw \"rectangle {2}\" /tmp/img.jpg {0}pattern{1}.jpg".format(pattern_path, k, rect))
                k += 1
    elif k == n:
        break
                

