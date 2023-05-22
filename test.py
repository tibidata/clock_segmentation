import numpy as np

from src import data_loader, model
import cv2

segmenter = model.Segmenter(image_path='/Users/tibortamas/PycharmProjects/clock_segmentation/faliora.jpg',
                            gauss_blur=True, kernel_size_g=(13, 13),
                            canny_threshold_1=260, canny_threshold_2=323)

gray = segmenter.bgr_to_grey()

blurred = segmenter.add_gauss_blur()

img = segmenter.remove_noise()

lines = segmenter.hough_lines()

hands = segmenter.cluster_points()

intersection = segmenter.calculate_time()

segmenter.show_detected_lines()
print(intersection)


