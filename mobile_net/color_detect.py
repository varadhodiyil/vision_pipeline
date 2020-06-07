# import the necessary packages
import numpy as np
import argparse
import cv2

# define the list of boundaries
boundaries = {
				# 'red': ([155, 56, 60], [190, 170, 197]),
				# # 'red': ([170, 100, 0], [180, 255, 255]),
				# 'black': ([0, 0, 27], [131, 65, 65]),
				# 'white':([73, 0, 178], [140, 21, 255]),
				# 'blue':([67,23,66], [116,106,255]),
				# 'silver':([117,0,112], [255,19,255])}
				'red': ([0, 100, 0], [10, 255, 255]),
				'red': ([160, 100, 75], [180, 190, 205]),
				'black': ([0, 0, 0], [120, 50, 50]),
				'white':([0, 0, 180], [10, 20, 255]),
				'blue':([80,40,125], [120,90,255]),
				'silver':([117,0,112], [255,19,255])}
				#'silver':([101,103,105], [187,189,191])}
				# 'silver':()}

def detect_color(image):			
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	max= 0
	#color = 'red'
	for key, value in boundaries.items():
		lower = np.array(value[0], dtype = "uint8")
		upper = np.array(value[1], dtype = "uint8")
		mask = cv2.inRange(hsv, lower, upper)
		pixel_count = np.sum(mask)
		#print(pixel_count)
		if (max < pixel_count):
			max = pixel_count
			#print(max)
			color = key
			#print(color)
			#break
	return color