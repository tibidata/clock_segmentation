import math

import cv2
import numpy as np
from sklearn.cluster import SpectralClustering

from src import data_loader


class Segmenter:

    def __init__(self, image_path: str,
                 gauss_blur: bool = False, kernel_size_g: tuple = (5, 5), sigma_x_g: int = 0,
                 canny_threshold_1: int = 210, canny_threshold_2: int = 300):
        self.hand_coord_list = None
        self.denoised_image = None
        self.lines = None
        self.edges = None
        self.gauss_blurred = None
        self.canny_threshold_2 = canny_threshold_2
        self.canny_threshold_1 = canny_threshold_1
        self.gray_image = None
        self.image_path = image_path
        self.gauss_blur = gauss_blur
        self.kernel_size_g = kernel_size_g
        self.sigma_x_g = sigma_x_g
        self.loader = data_loader.DataLoader(self.image_path)
        self.original_image = self.loader.load_image()

    def show_original_image(self):
        cv2.imshow('Original Image', self.original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bgr_to_grey(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        threshValue, binaryImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.gray_image = binaryImage

        return self.gray_image

    def add_gauss_blur(self):
        if self.gauss_blur:
            self.gauss_blurred = cv2.GaussianBlur(self.gray_image, self.kernel_size_g, self.sigma_x_g)

            return self.gauss_blurred
        else:
            pass

    def remove_noise(self):

        opIterations = 1

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        erodeImg = cv2.morphologyEx(self.gray_image, cv2.MORPH_ERODE, structuringElement, None, None, opIterations,
                                    cv2.BORDER_REFLECT101)
        self.denoised_image = cv2.morphologyEx(erodeImg, cv2.MORPH_DILATE, structuringElement, None, None, opIterations,
                                               cv2.BORDER_REFLECT101)
        return self.denoised_image

    def hough_lines(self):
        lines = cv2.HoughLinesP(self.denoised_image, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=100)

        self.lines = lines
        return self.lines

    def cluster_points(self):
        lines_reshape = np.reshape(self.lines, (6, 4))
        spect = SpectralClustering(n_clusters=2).fit_predict(lines_reshape)

        hand_1 = []
        hand_2 = []

        lines_list = lines_reshape.tolist()

        for i in range(len(lines_list)):
            if spect[i] == 0:
                hand_1.append(lines_list[i])
            else:
                hand_2.append(lines_list[i])

        hand1_coord = [int(x) for x in np.array(hand_1).mean(axis=0).tolist()]
        hand2_coord = [int(x) for x in np.array(hand_2).mean(axis=0).tolist()]

        self.hand_coord_list = [hand1_coord, hand2_coord]

        return self.hand_coord_list

    def calculate_time(self):
        hand_1_length = math.dist(self.hand_coord_list[0][:2], self.hand_coord_list[0][-2:])
        hand_2_length = math.dist(self.hand_coord_list[1][:2], self.hand_coord_list[1][-2:])

        second_hand_index = [hand_1_length, hand_2_length].index(max([hand_1_length, hand_2_length]))
        hour_hand_index = [hand_1_length, hand_2_length].index(min([hand_1_length, hand_2_length]))

        stacked = np.vstack([self.hand_coord_list[0][:2], self.hand_coord_list[0][-2:], self.hand_coord_list[1][:2],
                             self.hand_coord_list[1][-2:]])

        h = np.hstack((stacked, np.ones((4, 1))))

        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])

        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return float('inf'), float('inf')

        horizontal_line_coord = [x / z - 100, y / z, x / z + 100, y / z]

        self.hand_coord_list.append(horizontal_line_coord)
        return [x / z, y / z]

        #  return second_hand_index, hour_hand_index, hand_1_length, hand_2_length

    def show_detected_lines(self):

        X1 = []
        X2 = []
        Y1 = []
        Y2 = []

        # Store and draw the lines:
        for currentLine in self.hand_coord_list:
            print(currentLine)
            # First point:
            x1 = int(currentLine[0])
            y1 = int(currentLine[1])
            X1.append(x1)
            Y1.append(y1)

            # Second point:
            x2 = int(currentLine[2])
            y2 = int(currentLine[3])
            X2.append(x2)
            Y2.append(y2)

            # Draw the lines:
            cv2.line(self.original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("Lines", self.original_image)
            cv2.waitKey(0)
