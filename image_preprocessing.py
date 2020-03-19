import os
import matplotlib.pyplot as plt
import cv2 
import numpy as np

def pre_process(quantity):
	img_size = 100
	img_names = os.listdir('data')
	images = []
	for i in range(quantity):
		PATH = 'data/'+img_names[i]
		img = plt.imread(PATH)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (img_size,img_size))
		img = np.ndarray.flatten(img)
		img = img.reshape(img_size**2,)
		images.append(img)

	return np.array(images) 

def blur_images(quantity):
	size = 100
	img_names = os.listdir('data')
	blur_images = []
	for j in range(quantity):
		PATH = 'data/'+img_names[j]
		blur_img = plt.imread(PATH)
		blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
		blur_img = cv2.resize(blur_img, (size,size))
		blur_img = np.ndarray.flatten(blur_img)
		blur_img = cv2.medianBlur(blur_img,5)   
		blur_img = blur_img.reshape(size**2,)
		blur_images.append(blur_img)
	return np.array(blur_images)