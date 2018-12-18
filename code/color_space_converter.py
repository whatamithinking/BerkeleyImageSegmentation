from PIL import Image
import numpy as np
from skimage import io, color
import cv2
from img_util import wait_for_file_to_unlock

def rgb2rgb( img ):
	'''Return RGB image as RGB image numpy array

	Args:
		img (str,numpy.ndarray): Full path to the RGB image or numpy array
			presenting the image

	Returns:
		numpy.ndarray: numpy array containing image pixels in new color space

	'''
	if isinstance( img, str ):
		if not wait_for_file_to_unlock( img ):
			raise Exception( f'ERROR : IMAGE FILE LOCKED : {img}' )
		rgb_img = np.array( Image.open( img ) )
	else:
		rgb_img = img
	return rgb_img

def rgb2hsv( img ):
	'''Convert RGB image to HSV and return numpy array

	Args:
		img (str,numpy.ndarray): Full path to the RGB image or numpy array
			presenting the image

	Returns:
		numpy.ndarray: numpy array containing image pixels in new color space

	'''
	if isinstance( img, str ):
		if not wait_for_file_to_unlock( img ):
			raise Exception( f'ERROR : IMAGE FILE LOCKED : {img}' )
		rgb_img = Image.open( img )
	else:
		rgb_img = Image.fromarray( img, mode='RGB' )
	hsv_img = rgb_img.convert( 'HSV' )
	return np.array( hsv_img )

def rgb2lab( img ):
	'''Convert RGB to LAB and return numpy array

	Args:
		img (str,numpy.ndarray): Full path to the RGB image or numpy array
			presenting the image

	Returns:
		numpy.ndarray: numpy array containing image pixels in new color space

	'''
	if isinstance( img, str ):
		if not wait_for_file_to_unlock( img ):
			raise Exception( f'ERROR : IMAGE FILE LOCKED : {img}' )
		rgb_img = io.imread( img )
	else:
		rgb_img = img
	lab_img = color.rgb2lab( rgb_img )
	return np.array( lab_img )

def rgb2luv( img ):
	'''Convert RGB to LUV and return numpy array

	Args:
		img (str,numpy.ndarray): Full path to the RGB image or numpy array
			presenting the image

	Returns:
		numpy.ndarray: numpy array containing image pixels in new color space

	'''
	if isinstance( img, str ):
		if not wait_for_file_to_unlock( img ):
			raise Exception( f'ERROR : IMAGE FILE LOCKED : {img}' )
		rgb_img = np.array( Image.open( img ) )
	else:
		rgb_img = img
	luv_img = cv2.cvtColor( rgb_img, cv2.COLOR_RGB2LUV )
	return luv_img

# SUPPORTED COLOR SPACES
COLOR_MAPPINGS = {
	'RGB'  : rgb2rgb
	,'HSV' : rgb2hsv	
	,'LAB' : rgb2lab	
	,'LUV' : rgb2luv	
}