from cv2 import cv2
import numpy as np

def remove_background(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    img_copy = img[y: y + h, x: x + w]

    return img_copy


path = "../Images/Training/board/IMG_20210425_131243.jpg"

img = cv2.imread(path)

img2 = remove_background(img, threshold=60)
cv2.imshow("img2", img2)
rows, cols = img2.shape[:2]
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)


cnts = cv2.findContours(edges,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[0]
cnt = sorted(cnts, key=cv2.contourArea)[-1]
print(cnts[-1][0][0])
cv2.drawContours(img2, cnts, -1, (255, 0, 0), 2)
cv2.imshow("img", img2)

src_points = np.float32([cnts[-1][0][0], cnts[-2][0][0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0],[0,rows-1], [cols-1,rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img2, projective_matrix, (cols,rows))
cv2.imshow("transformed", img_output)

# cv2.imshow("img", img2)
# edges = cv2.Canny(img2, 100, 200)
# cv2.imshow("edges", edges)

cv2.waitKey(0)
