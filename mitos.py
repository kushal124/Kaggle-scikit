import glob
import os
from scipy import misc

#Generate file list from directory
file_list = glob.glob("*.tiff")


img = {}

#Read all images as ndarray in a dict
for file in file_list:
	img[os.path.splitext(file)[0]] = misc.imread(file)

