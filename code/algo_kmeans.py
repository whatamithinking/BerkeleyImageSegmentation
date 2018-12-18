from sklearn.cluster import KMeans
from seg_util import *
from img_util import *
from constants import *
from color_space_converter import *

K_MEANS_DEFAULT_MAX_ITER = 50
K_MEANS_VAR_DICT = {
	'n_clusters'	: [6,7,8,9,10]
	,'max_iter'		: [5,10,20,30,40,50]
	,'color_space'	: ['RGB','HSV','LUV','LAB']
}

def segment( img, n_clusters=DEFAULT_N_CLUSTERS, max_iter=K_MEANS_DEFAULT_MAX_ITER, color_space=DEFAULT_COLOR_SPACE ):
	
	original_img_shape = img.shape
	color_cvt_func = COLOR_MAPPINGS[ color_space ]
	img_array = color_cvt_func( img )
	img_2D_array = get_img_2D( img_array )

	kmeans=KMeans(
		random_state=RANDOM_SEED
		,n_clusters=n_clusters # number of clusters we expect are in data
		,max_iter=max_iter # max number of iterations before we force algo to stop whether it converged or not
		,n_init=1	# number of runs with diff cluster centers to start with
		,init='random' # use random cluster centers
		,algorithm='full' # use classic kmeans algo
	)
	seg_array = kmeans.fit( img_2D_array )
	
	labels_list = np.array( seg_array.labels_ )
	labels_array = labels_list.reshape((original_img_shape[0],original_img_shape[1])) # reshape to 2D array to match up with image pixels

	return labels_array

