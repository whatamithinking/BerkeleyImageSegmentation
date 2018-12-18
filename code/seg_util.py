import csv, os, glob, itertools
from datetime import datetime
import numpy as np
from constants import *
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt

def load_segmentation( filepath ):
	'''Load segmentation file into memory as numpy array

	Args:
		filepath (str): full path to the segmentation file

	Returns:
		numpy.ndarray: an array of same width and height as segmented image,
			with each cell value containing a segment label

	'''
	with open( filepath, 'r' ) as ofile:
		reader = csv.reader( ofile, delimiter=' ' )
		
		# LOAD HEADER INFO
		width = 0
		height = 0
		for row in reader:	# this approach required, because sometimes part of header missing. not always 11 rows
			if row[0] == 'data':
				break
			elif row[0] == 'width':
				width = int(row[1])
			elif row[0] == 'height':
				height = int(row[1])

		# LOAD ACTUAL SEGMENTATION DATA
		seg_array = np.empty((height,width)) # create empty array to hold segmentation
		for row in reader:
			label = int(row[0]) # label applied to this small section of image
			row_index = int(row[1]) # row in image
			start_col = int(row[2]) # first column where label appears in row continuously
			stop_col = int(row[3]) # last column where label appears in row continuously
			seg_array[row_index,start_col:stop_col+1] = int(label)

		return seg_array

def save_segmentation( seg_array, filepath, npr=None, imagename='0', user='0', is_gray=False, is_inverted=False, is_flipflop=False ):
	'''Save segmentation to file, in Berkeley .seg format

	Args:
		seg_array (numpy.ndarray): numpy array containing the segmentation labels
		filepath (str): full path to the segmentation file

	Returns:
		Nothing

	'''
	with open( filepath, 'w' ) as ofile:
		writer = csv.writer( ofile, delimiter=' ', lineterminator='\n' )
		
		# WRITE HEADER INFO
		writer.writerow( ['format','ascii','cr'] )
		weekday = datetime.now().strftime( '%a' )
		month = datetime.now().strftime( '%b' )
		day = datetime.now().strftime( '%d' )
		time = datetime.now().strftime( '%H:%M:%S' )
		year = datetime.now().strftime( '%Y' )
		writer.writerow( ['date', weekday, month, day, time, year ] )
		writer.writerow( ['image', imagename ] )
		writer.writerow( ['user', user ] )
		width = seg_array.shape[1]
		writer.writerow( ['width', width ] )
		height = seg_array.shape[0]
		writer.writerow( ['height', height ] )
		segment_count = np.amax( seg_array )
		writer.writerow( ['segments', segment_count ] )
		int_is_gray = int( is_gray )
		writer.writerow( ['gray', int_is_gray ] )
		int_is_inverted = int( is_inverted )
		writer.writerow( ['invert', int_is_inverted ] )
		int_is_flipflop = int( is_inverted )
		writer.writerow( ['flipflop', int_is_flipflop ] )
		if npr != None: # include normalized probabilistic rand index if given
			writer.writerow( ['npr', npr] )
		writer.writerow( ['data'] )

		# WRITE ACTUAL SEGMENTATION DATA (label,row index,start col, stop col)
		for i, row in enumerate( seg_array ):
			istart = 0
			istop = 0
			for label, label_group in itertools.groupby( row ):
				pixels_with_label = list( label_group )
				istop += len( pixels_with_label ) - 1
				writer.writerow( [ label, i, istart, istop ] ) # label, row index, start col, stop col
				istart += len( pixels_with_label ) # increment to start of next section
				istop += 1

def display_segmentation( seg_array ):
	'''Plot/show the segmentation array with unique color for each label.

	Args:
		seg_array (numpy.ndarray): The segmentation array, 2D, with the labels

	Returns:
		Nothing

	'''
	segment_count = np.amax(seg_array)+1
	color_map = get_cmap('hsv',segment_count)
	plt.imshow(seg_array,cmap=color_map)
	plt.show()

def get_berkeley_seg_filepaths( img_name ):
	'''Find .seg files for given image name from Berkeley segmentations

	Args:
		img_name (str): name of the image

	Returns:
		iterable: generator of .seg filepaths for this image name

	'''
	search_filepath = BERKELEY_SEG_DIR + '\\**\\%s.seg' % img_name
	for gt_seg_filepath in glob.iglob( search_filepath ):
		yield gt_seg_filepath

def update_summary( img_name, algo_name, var_name, var_val, var_range, npr ):
	summary_filepath = ALGO_SEG_DIR + f'\\{algo_name}\\{var_name}\\summary.csv'
	fieldnames = [ 'img_name' ] + [ x for x in var_range ]
	
	# CREATE FILE IF IT DOESN'T EXIST
	if not os.path.exists( summary_filepath ):
		with open( summary_filepath, 'w' ) as ofile:
			writer = csv.DictWriter( ofile, delimiter=',', lineterminator='\n', fieldnames=fieldnames )
			writer.writeheader()

	# FIND ROW TO UPDATE
	summary_data = None
	row_to_update = -1
	with open( summary_filepath, 'r' ) as ofile:
		reader = csv.reader( ofile, delimiter=',' )
		next(reader,None)
		summary_data = [ row for row in reader ]
	for i, row in enumerate( summary_data ):
		if row[ fieldnames.index( 'img_name' ) ] == img_name:
			row_to_update = i
			break

	# ADD IMG NAME IF NOT IN FILE YET
	if row_to_update == -1:
		empty_row = [None]*len(fieldnames)
		empty_row[ fieldnames.index( 'img_name' ) ] = img_name
		empty_row[ fieldnames.index( var_val ) ] = npr
		summary_data.append( empty_row )
	else:
		summary_data[ row_to_update ][ fieldnames.index( var_val ) ] = npr

	# UPDATE IMG NAME / VAR VAL WITH NPR
	with open( summary_filepath, 'w' ) as ofile:
		writer = csv.writer( ofile, delimiter=',', lineterminator='\n' )
		writer.writerow( fieldnames )
		writer.writerows( summary_data )

def get_segmented_img_names( algo_name, var_name ):
	summary_filepath = ALGO_SEG_DIR + f'\\{algo_name}\\{var_name}\\summary.csv'
	if not os.path.exists( summary_filepath ):
		return []
	summary_data = None
	with open( summary_filepath, 'r' ) as ofile:
		reader = csv.reader( ofile, delimiter=',' )
		next(reader,None)
		summary_data = [ row for row in reader ]
	for row in summary_data:
		if all( x not in (None,'') for x in row ): # make sure all var values have been completed
			yield row[0]
	