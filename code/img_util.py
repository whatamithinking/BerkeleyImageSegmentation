import numpy as np
from PIL import Image
from constants import *
import time

def wait_for_file_to_unlock( filepath, timeout=10 ):
	start_time = time.process_time()
	while time.process_time() - start_time <= timeout:
		try:
			if open(filepath,'a',8):
				return True
		except:
			pass
		time.sleep( 0.01 )
	return False

def get_img_names():
	for f in os.listdir( BERKELEY_IMG_DIR ):
		yield f.split('.')[0]

def get_img_filepath( img_name ):
	'''Return full path for given image name
	'''
	return BERKELEY_IMG_DIR + "\\" + img_name + ".jpg"

def get_img_shape( img_name ):
	'''Get shape of berkeley image
	'''
	return np.array( Image.open( get_img_filepath( img_name ) ) ).shape

def get_img_2D( img_array ):
	'''Convert the image, in whatever color space, to a 2D array.
	'''
	feature_count = img_array.shape[2] # count number of features in image
	img_2D = img_array.reshape( -1, feature_count ) # convert to 2D version, b/c PCA con only handle 2D
	return img_2D