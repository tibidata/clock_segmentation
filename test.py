import numpy as np

from src import data_loader, model
import cv2

segmenter = model.Segmenter(image_path='/Users/tibortamas/PycharmProjects/clock_segmentation/faliora.jpg')

gray = segmenter.bgr_to_grey()

blurred = segmenter.add_gauss_blur()

img = segmenter.remove_noise()

cv2.imshow('image', img)
cv2.waitKey(0)

lines = segmenter.hough_lines()

hands = segmenter.cluster_points()

angles = segmenter.calculate_time()

#  segmenter.show_detected_lines()
print(angles)


