import numpy as np
import argparse
import cv2


boundaries = {#'red': ([155, 56, 60], [190, 170, 197]),
				'red': ([0, 100, 0], [10, 255, 255]),
				'red': ([170, 100, 0], [180, 255, 255]),
				# 'black': ([0, 0, 27], [131, 65, 65]),
				'black': ([0, 0, 0], [180, 255, 30]),
				'white':([0, 0, 200], [180, 255, 255]),
				# 'white':([73, 0, 178], [140, 21, 255]),
				# 'blue':([67,23,66], [116,106,255]),
				'blue':([110,50,50], [130,255,255])
				}
				# 'silver':([117,0,112], [255,19,255])}
			
class ColorClassifier:
	def __init__(self):
		pass

	def detect_color(self, image):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		max= 0
		color = 'red'
		for key, value in boundaries.items():
			lower = np.array(value[0], dtype = "uint8")
			upper = np.array(value[1], dtype = "uint8")
			mask = cv2.inRange(hsv, lower, upper)
			pixel_count = np.sum(mask)
			# print(pixel_count)
			if (max < pixel_count):
				max = pixel_count
				color = key
		# print(key)
		return color

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())
# # load the image

# image = cv2.imread('./images/train-val/hatchback/images4.jpg')
# #Red, Black, White, Blue, Silver
# boundaries = {'red': ([0, 100, 0], [10, 255, 255]),
# 				'red': ([170, 100, 0], [180, 255, 255]),py
# 				'black': ([0, 0, 0], [180, 255, 30]),
# 				'white':([0, 0, 200], [180, 255, 255]),
# 				'blue':([110,50,50], [130,255,255])}
# 				# 'silver':()}
# print(boundaries.values())
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# max= 0
# color = 'red'
# for key, value in boundaries.items():
# 	lower = np.array(value[0], dtype = "uint8")
# 	upper = np.array(value[1], dtype = "uint8")
# 	mask = cv2.inRange(hsv, lower, upper)
# 	pixel_count = np.sum(mask)
# 	print(pixel_count)
# 	if (max < pixel_count):
# 		max = pixel_count
# 		color = key

# print(color, max)
