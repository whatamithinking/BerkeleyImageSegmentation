from seg_util import *
from img_util import *
from sklearn.metrics import adjusted_rand_score

def avg_npr( seg_array, img_name, verbose=False ):
	'''Calculate average adjusted rand index (or average NPR normalized prob. rand index) using all available Berkeley segmentations for img

	Args:
		seg_array (numpy.ndarray): NEW segmentation using NEW algo, to be compared against ground truth segmentations
		img_name (str): The name of the image in the Berkeley dataset

	Returns:
		average_npr (float): The average npr across all available segmentations of the image

	'''

	img_filepath = get_img_filepath( img_name )

	# FLATTEN SEGMENTATION GIVEN
	seg_list = seg_array.flatten().tolist()

	# FIND ALL GROUND TRUTH SEGMENTATION FILES FOR THIS IMAGE
	npr_list = []
	for gt_seg_filepath in get_berkeley_seg_filepaths( img_name ):
		if verbose: 
			print(gt_seg_filepath)

		# LOAD GROUND TRUTH SEGMENTATION
		ground_truth_array = load_segmentation( gt_seg_filepath )
		ground_truth_list = ground_truth_array.flatten().tolist()

		# FIND RAND INDEX
		npr = adjusted_rand_score( ground_truth_list, seg_list )
		npr_list.append(npr)

	average_npr = sum(npr_list)/len(npr_list)
	return average_npr





