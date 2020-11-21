import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

st.title("Image Processor")
st.write("Turn your images grayscale, black and white, or negative!")
st.write("Use the widgets to adjust your image")

rose2 = 'roses.jpg'	
user_input = st.text_input("Upload an image to get started", rose2)

st.write("Original image:")
orig_image = cv2.imread(user_input)
orig_resized = cv2.resize(orig_image, (200, int(200*orig_image.shape[0]/orig_image.shape[1])))
orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
st.image(orig_resized)

def bw_img(impath, blur=True, gauss=11, thresh_type='Constant', thresh=70, invert='Normal'):
	img = cv2.imread(impath, 0)
    
    # Clean up image using Gaussian blur
	if blur:
		img = cv2.GaussianBlur(img, (gauss, gauss), 0)
	else:
		img = img
	
	# Invert binarize the image
	if thresh_type == 'Constant':
		ret, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
	elif thresh_type == 'Adaptive':
		mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh, thresh)
	else:
		mask = img
	resized = cv2.resize(mask, (600, int(600*mask.shape[0]/mask.shape[1])))
	if invert == 'Inverted':
		resized = 255-resized
	st.image(resized)
	return mask
	
def grayscale_img(impath, blur=True, gauss=11):
	img = cv2.imread(impath, 0)

	# Clean up image using Gaussian blur
	if blur:
		img = cv2.GaussianBlur(img, (gauss, gauss), 0)
	else:
		img = img

	resized = cv2.resize(img, (600, int(600*img.shape[0]/img.shape[1])))
	st.image(resized)
	return img

def negative_img(impath, gray=False, blur=True, gauss=11):
	if gray:
		img = cv2.imread(impath, 0)
	else:
		img = cv2.imread(impath)
	if blur:
		img = cv2.GaussianBlur(img, (gauss, gauss), 0)
	else:
		img = img
	img = 255-img
	resized = cv2.resize(img, (600, int(600*img.shape[0]/img.shape[1])))
	st.image(resized)
	return img

# Set up widgets
channel = st.sidebar.radio("", ['Grayscale', 'Negative-grayscale', 'B&W', 'Negative-color'])
blur = st.sidebar.radio("Blur?", [True, False])

# Widget options
if blur:
	gauss = st.sidebar.selectbox("blur", [1,3,5,7,9,11,13,15,17,19])
else:
	gauss = None

if channel == 'B&W':
	thresh_type = st.sidebar.radio("Threshold type", ['Constant', 'Adaptive'])
	invert = st.sidebar.radio("Invert", ['Normal', 'Inverted'])
	if thresh_type == 'Constant':
		thresh = st.sidebar.slider("Threshold", 40,200)
	elif thresh_type == 'Adaptive':
		thresh = st.sidebar.selectbox("Threshold", [3,5,7,9,11,13,15,17,19,21])
	img = bw_img(user_input, blur=blur, gauss=gauss, thresh_type =thresh_type, thresh=thresh, invert=invert)
elif channel == 'Grayscale':
	img = grayscale_img(user_input, blur=blur, gauss=gauss)
elif channel == 'Negative-color':
	img = negative_img(user_input, blur=blur, gauss=gauss)
elif channel == 'Negative-grayscale':
	img = negative_img(user_input, gray=True, blur=blur, gauss=gauss)


save_path = st.text_input("Enter the filepath for saving your image", '')
if st.button('Download'):
	cv2.imwrite(save_path, img)
	st.write("Saved image as:", save_path)