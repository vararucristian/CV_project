import numpy as np
from cv2 import cv2
class Board:

    def __init__(self):
        pass

    def cutImage(self):
        return img[self.y:self.y + self.h, self.x: self.x + self.w]


    def detectBoardCorners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 7))
        bottom_w = abs(corners[0, 0, 0] - corners[1, 0, 0])
        bottom_h = abs(corners[7, 0, 1] - corners[1, 0, 1])
        bottom_h += 0.1 * bottom_h
        bottom_w += 0.2 * bottom_w

        top_w = abs(corners[43, 0, 0] - corners[42, 0, 0])
        top_h = abs(corners[35, 0, 1] - corners[42, 0, 1])
        top_h -= 0.1 * top_h
        top_w -= 0.2 * top_w

        bottom_right_corner = (int(corners[0, 0, 0] + bottom_w), int(corners[0, 0, 1] + bottom_h))
        bottom_left_corner = (int(corners[6, 0, 0] - bottom_w), int(corners[6, 0, 1] + bottom_h))
        top_right_corner = (int(corners[42, 0, 0] + top_w), int(corners[42, 0, 1] - top_h))
        top_left_corner = (int(corners[48, 0, 0] - top_w), int(corners[48, 0, 1] - top_h))

        return np.float32([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner])


    def setBoardPositions(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        ksize = (5, 5)
        edges = cv2.blur(edges, ksize)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        self.x, self.y, self.w, self.h = cv2.boundingRect(cnt)


    def transposeBoard(self, img, src_points):
        rows, cols, _ = img.shape
        dst_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

        return img_output


if __name__ == "__main__":
    path = "../Images/test/IMG_20210516_155207.jpg"
    # path = "../Images/Training/board/IMG_20210425_131243.jpg"
    img = cv2.imread(path)
    img = cv2.bitwise_not(img)

    board = Board()
    board.setBoardPositions(img)
    img = board.cutImage()
    points = board.detectBoardCorners(img);
    cv2.imshow("img", board.transposeBoard(img, points))

    cv2.waitKey(0)
