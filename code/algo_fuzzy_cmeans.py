import skfuzzy as fuzz
from seg_util import *
from img_util import *
from constants import *
import time
from color_space_converter import *

from seg_eval import *

FUZZY_CMEANS_DEFAULT_M=2
FUZZY_CMEANS_DEFAULT_ERROR=0.005
FUZZY_CMEANS_VAR_DICT = {
	'n_clusters'	: [6,7,8,9,10]
	,'max_iter'		: [6,7,8,9,10]
	,'color_space'	: ['RGB','HSV','LUV','LAB']
	,'m'			: [0.01,0.5,0.8,0.9,0.95,0.99,1.01,1.1,1.2,1.5,2,5]
}

def segment( img, n_clusters=DEFAULT_N_CLUSTERS, max_iter=DEFAULT_MAX_ITER, color_space=DEFAULT_COLOR_SPACE, m=FUZZY_CMEANS_DEFAULT_M ):
	
	original_img_shape = img.shape
	color_cvt_func = COLOR_MAPPINGS[ color_space ]
	img_array = color_cvt_func( img )
	img_2D_array = get_img_2D( img_array )

	cluster_centers,fuzzy_labels,_,_,_,_,_=fuzz.cluster.cmeans(
		data=img_2D_array.T # transpose image data to match fuzz expected format
		,seed=RANDOM_SEED
		,c=n_clusters # number of clusters we expect are in data
		,maxiter=max_iter # max number of iterations before we force algo to stop whether it converged or not
		,m=m # array exponentiation applied to membership matrix at each iteration
		,error=FUZZY_CMEANS_DEFAULT_ERROR # if change from previous membership value less than this, as percent, then stopping criteria met
	)
	
	labels_list = np.argmax(fuzzy_labels,axis=0).T
	labels_array = np.array(labels_list).reshape((original_img_shape[0],original_img_shape[1]))		

	return labels_array

# img = np.array( Image.open( get_img_filepath( '100075' ) ) )
# # seg_array = segment( img, m=0.5 )
# # npr = avg_npr( seg_array, '100075' )  
# # print(npr)
# seg_array = segment( img, m=1.0000001 )
# npr = avg_npr( seg_array, '100075' )  
# print(npr)