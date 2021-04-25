from cv2 import cv2

path = "../Images/Training/board/IMG_20210425_131243.jpg"

img = cv2.imread(path)
# cv2.imshow("image", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("edges", edges)
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(cnts, key=cv2.contourArea)[-1]

cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
cv2.imshow("img", img)

cv2.waitKey(0)
