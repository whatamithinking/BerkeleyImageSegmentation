from algo_kmeans import segment as kmeans_segment
from algo_fuzzy_cmeans import segment as fuzzy_cmeans_segment
from color_space_converter import COLOR_MAPPINGS
from seg_util import get_segmented_img_names
from img_util import get_img_names
from tqdm import tqdm
from seg_eval import avg_npr
from constants import *
from seg_util import *
from img_util import get_img_filepath, wait_for_file_to_unlock
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import fire

ALGO_NAMES_DICT = {
	'kmeans'  			: (kmeans_segment,KMEANS_DIR)
	,'fuzzy_cmeans'		: (fuzzy_cmeans_segment,FUZZY_CMEANS_DIR)
}

def __segment( seg_func, seg_dir, img_name, img, var_name, var_val ):
	var_dict = { var_name : var_val }
	seg_array = seg_func( img, **var_dict )
	average_npr = avg_npr( seg_array, img_name )
	var_val_dir = f'{seg_dir}\\{var_name}\\{var_val}'
	if not os.path.exists( var_val_dir ):
		os.mkdir( var_val_dir )
	seg_filepath = var_val_dir + f'\\{img_name}.seg'
	save_segmentation( seg_array, seg_filepath, npr=average_npr, imagename=img_name )
	return var_val, average_npr

def seg_looper( algo_name, var_name, var_range ):
	
	seg_func, seg_dir = ALGO_NAMES_DICT[ algo_name ]
	segmented_img_names = list( get_segmented_img_names( algo_name, var_name ) ) # get list of image names already segmented, in case previously interrupted
	img_names = [ x for x in get_img_names() if not x in segmented_img_names ]
	var_val_list = list( var_range )
	
	pool = ProcessPoolExecutor( max_workers=len(var_val_list) )
	unit = algo_name + ' ' + var_name + ' Images'
	for img_name in tqdm( img_names, unit=unit, leave=True ):
		
		img_filepath = get_img_filepath( img_name )
		if not wait_for_file_to_unlock( img_filepath ):
			raise Exception( f'ERROR : IMAGE FILE LOCKED : {img_filepath}' )
		img = np.array( Image.open( img_filepath ) )

		# BUILD SET OF PARALLEL JOBS
		futures = []
		for var_val in var_val_list:
			futures.append( pool.submit( __segment, seg_func, seg_dir, img_name, img.copy(), var_name, var_val ) )
		
		# GET RESULTS FROM PARALLEL JOBS
		unit=algo_name + ' ' + var_name.upper()
		with tqdm( total=len(futures), unit=unit, leave=False ) as pbar:
			for future in as_completed( futures ):
				var_val, npr = future.result()
				update_summary( img_name, algo_name, var_name, var_val, var_val_list, npr )
				pbar.update()		

def __optimal_segment( seg_func, seg_dir, img_name, **kwargs ):
	img_filepath = get_img_filepath( img_name )
	if not wait_for_file_to_unlock( img_filepath ):
		raise Exception( f'ERROR : IMAGE FILE LOCKED : {img_filepath}' )
	img = np.array( Image.open( img_filepath ) )
	seg_array = seg_func( img, **kwargs )
	average_npr = avg_npr( seg_array, img_name )
	seg_filepath = f'{seg_dir}\\{img_name}.seg'
	save_segmentation( seg_array, seg_filepath, npr=average_npr, imagename=img_name )
	return img_name, average_npr

def optimal_seg_runner( algo_name, **kwargs ):
	
	if algo_name == 'kmeans':
		seg_dir = KMEANS_OPTIMAL_SEG_DIR
		seg_func = kmeans_segment
	elif algo_name == 'fuzzy_cmeans':
		seg_dir = FUZZY_CMEANS_OPTIMAL_SEG_DIR
		seg_func = fuzzy_cmeans_segment
	
	with open( f'{seg_dir}\\summary.csv', 'w' ) as ofile:
		writer = csv.writer( ofile, delimiter=',', lineterminator='\n' )
		writer.writerow( ['img_name','npr'] )

	futures = []
	pool = ProcessPoolExecutor( max_workers=10 )
	for img_name in get_img_names():
		futures.append( pool.submit( __optimal_segment, seg_func, seg_dir, img_name, **kwargs ) )
		
	# GET RESULTS FROM PARALLEL JOBS
	with tqdm( total=len(futures), unit='Images', leave=True ) as pbar:
		for future in as_completed( futures ):
			img_name, npr = future.result()
			with open( f'{seg_dir}\\summary.csv', 'a' ) as ofile:
				writer = csv.writer( ofile, delimiter=',', lineterminator='\n' )
				writer.writerow( [img_name,npr] )
			pbar.update()

if __name__ == '__main__':
	fire.Fire()