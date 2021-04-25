import cv2
import os
path = '../Images/Training/board'
resized_path = '../Images/Training/board'

for root, dir, files in os.walk(path):
    for file in files:
        img = cv2.imread(os.path.join(root,file))
        dim = (1000, 751)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print(files.index(file) + 1, "/", len(files))
        cv2.imwrite(os.path.join(resized_path, file), resized)