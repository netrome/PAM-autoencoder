import PIL
import glob
import sys
import os

place = os.getcwd()
lfw_path = place + "/lfw-deepfunneled/"
out_path = place + "/small_pics/"

k = 0
n = 100000
for path, dirs, files in os.walk(lfw_path):
    if len(files) > 0 and k < n:
        for i in files:
            image_path = path + "/" + i
            os.system("convert -resize 64x64 {0} {1}out{2}.jpg".format(image_path, out_path, k))

            k += 1

