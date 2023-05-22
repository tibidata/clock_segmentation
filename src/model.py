import math
import warnings

import cv2
import numpy as np
from sklearn.cluster import SpectralClustering

from src import data_loader


class Segmenter:

    def __init__(self, image_path: str,
                 gauss_blur: bool = False, kernel_size_g: tuple = (5, 5), sigma_x_g: int = 0):

        self.image_path = image_path
        self.gauss_blur = gauss_blur
        self.kernel_size_g = kernel_size_g
        self.sigma_x_g = sigma_x_g

        # Loading image
        warnings.filterwarnings('ignore')
        self.loader = data_loader.DataLoader(self.image_path)
        self.original_image = self.loader.load_image()

        # Variables which going to be calculated later

        self.hand_coord_list = None
        self.denoised_image = None
        self.lines = None
        self.edges = None
        self.gauss_blurred = None
        self.gray_image = None

    def show_original_image(self):
        """
        Function to show original image
        :return: None
        """
        cv2.imshow('Original Image', self.original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bgr_to_grey(self):
        """
        Convert color image to greyscale image
        :return: grey scale image
        """

        #  Apply greyscale converter and binary converter to original image

        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        threshValue, binaryImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self.gray_image = binaryImage

        return self.gray_image

    def add_gauss_blur(self):
        """
        Optional function to apply Gaussian blur to the image, not used during the algorithm
        :return: blurred image
        """
        if self.gauss_blur:
            self.gauss_blurred = cv2.GaussianBlur(self.gray_image, self.kernel_size_g, self.sigma_x_g)

            return self.gauss_blurred
        else:
            pass

    def remove_noise(self):
        """
        Removes noise from the image
        :return: denoised image
        """

        opIterations = 1

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Apply erosion to the image

        erodeImg = cv2.morphologyEx(self.gray_image, cv2.MORPH_ERODE, structuringElement, None, None, opIterations,
                                    cv2.BORDER_REFLECT101)

        # Apply dilation on eroded image

        self.denoised_image = cv2.morphologyEx(erodeImg, cv2.MORPH_DILATE, structuringElement, None, None, opIterations,
                                               cv2.BORDER_REFLECT101)
        return self.denoised_image

    def hough_lines(self):
        """
        Applies probabilistic hough lines algorithm
        :return: ndarray : array of lines detected on the image, [x1, y1, x2, y2]
        where x1, y1 are the start points of the line
        """
        lines = cv2.HoughLinesP(self.denoised_image, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=100)

        self.lines = lines
        return self.lines

    def cluster_points(self):
        """
        Clustering the lines obtained with the hough transformation to get the coordinates of the 2 hands
        :return: list: list of the coordinates of the hands
        """

        # Reshaping array to the necessary shape
        lines_reshape = np.reshape(self.lines, (6, 4))

        # Using spectral clustering to cluster the lines
        spect = SpectralClustering(n_clusters=2).fit_predict(lines_reshape)

        hand_1 = []
        hand_2 = []

        lines_list = lines_reshape.tolist()

        # Separating the lines to get the ones which belong to the same hand

        for i in range(len(lines_list)):
            if spect[i] == 0:
                hand_1.append(lines_list[i])
            else:
                hand_2.append(lines_list[i])

        # Getting the average of the lines in the same cluster to obtain 1 line

        hand1_coord = [int(x) for x in np.array(hand_1).mean(axis=0).tolist()]
        hand2_coord = [int(x) for x in np.array(hand_2).mean(axis=0).tolist()]

        self.hand_coord_list = [hand1_coord, hand2_coord]

        return self.hand_coord_list

    def calculate_time(self):
        """
        Calculates the time
        :return:str: string of the time
        """

        # Calculating the length of the hands to decide which is the hour and which is the minute
        hand_1_length = math.dist(self.hand_coord_list[0][:2], self.hand_coord_list[0][-2:])
        hand_2_length = math.dist(self.hand_coord_list[1][:2], self.hand_coord_list[1][-2:])

        # Setting the smaller hand as the hour hand and the longer as the minute
        minute_hand_index = [hand_1_length, hand_2_length].index(max([hand_1_length, hand_2_length]))
        hour_hand_index = [hand_1_length, hand_2_length].index(min([hand_1_length, hand_2_length]))

        stacked = np.vstack([self.hand_coord_list[0][:2], self.hand_coord_list[0][-2:], self.hand_coord_list[1][:2],
                             self.hand_coord_list[1][-2:]])

        # Calculating the intersection of the hands
        h = np.hstack((stacked, np.ones((4, 1))))

        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])

        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return float('inf'), float('inf')

        vertical_line_coord = [x / z, y / z, x / z, y / z - 100]

        vect_hand1 = np.array(self.hand_coord_list[0][:2]) - np.array(self.hand_coord_list[0][-2:])
        vect_hand2 = np.array(self.hand_coord_list[1][:2]) - np.array(self.hand_coord_list[1][-2:])

        # Creating a vertical line from the intersection
        vect_vertical = np.array(vertical_line_coord[:2]) - np.array(vertical_line_coord[-2:])

        angles_list = []
        vect_hand_list = [vect_hand1, vect_hand2]

        # Calculating the angles between the vertical line and the hands
        for vect in vect_hand_list:
            alpha = math.degrees(math.acos(np.dot(vect, vect_vertical) /
                                           (np.sqrt(vect.dot(vect)) * np.sqrt(
                                               vect_vertical.dot(vect_vertical)))))

            # Handling the case when one of the hands is on the left of the vertical line
            if np.dot(vect, vect_vertical) < 0:
                angles_list.append(alpha + 180)
            else:
                angles_list.append(alpha)

        # Calculating the time based on the angles
        minutes = round(angles_list[minute_hand_index] / 6)
        hours = round(angles_list[hour_hand_index] / 30)

        return 'The time is ' + str(hours) + ' hours and ' + str(minutes) + ' minutes'

    def show_detected_lines(self):
        """
        Shows the detected lines on the original image
        :return: None
        """

        X1 = []
        X2 = []
        Y1 = []
        Y2 = []

        # Store and draw the lines:
        for currentLine in self.hand_coord_list:
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

    def segment(self):
        """
        Runs the image pre-processing steps, hough lines transformation, clustering and calculates time
        :return: str: The string of the current time
        """
        self.bgr_to_grey()
        self.remove_noise()
        self.hough_lines()
        self.cluster_points()
        time = self.calculate_time()

        return time
