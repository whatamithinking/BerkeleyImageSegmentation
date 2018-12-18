from constants import *
from img_util import get_img_filepath
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from seg_util import *
from tqdm import tqdm
import csv
from algo_kmeans import K_MEANS_VAR_DICT
from algo_fuzzy_cmeans import FUZZY_CMEANS_VAR_DICT

algo_names = [ 
	( 'kmeans', K_MEANS_VAR_DICT )
	,( 'fuzzy_cmeans', FUZZY_CMEANS_VAR_DICT )
]

def one_off_seg_img_compare( seg_dir, plt_title, title, img_names=[] ):
	
	FONT_SIZE = 20

	f,ax = plt.subplots(len(img_names),3,figsize=[24,24])
	
	for r, img_name in tqdm( enumerate( img_names ), unit='Images', leave=False ):
		
		# SHOW ORIGINAL IMAGE
		img_filepath = get_img_filepath( img_name )
		img_array = np.array( Image.open( img_filepath ) )
		ax[r,0].imshow( img_array )
		ax[r,0].set_title( img_name,fontweight="bold", size=FONT_SIZE )
		ax[r,0].axis('off')

		# SHOW AN IDEAL SEGMENTATION
		seg_filepath = list( get_berkeley_seg_filepaths( img_name ) )[0]
		seg_array = load_segmentation( seg_filepath )
		segment_count = np.amax(seg_array)+1
		color_map = get_cmap('hsv',segment_count)
		ax[r,1].imshow( seg_array, cmap=color_map )
		ax[r,1].set_title( 'IDEAL SEG',fontweight="bold", size=FONT_SIZE )
		ax[r,1].axis('off')

		# LOAD SEG ARRAY FROM MEMORY
		img_seg_filepath = f"{seg_dir}\\{img_name}.seg"
		seg_array = load_segmentation( img_seg_filepath )
		segment_count = np.amax(seg_array)+1
		color_map = get_cmap('hsv',segment_count)
		
		# DISPLAY SEG ARRAY
		ax[r,2].imshow( seg_array, cmap=color_map )
		ax[r,2].set_title( plt_title,fontweight="bold", size=FONT_SIZE )
		ax[r,2].axis('off')

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.2,wspace=0)
	save_filepath = f"{REPORT_DIR}\\images_{title}.jpg"
	plt.savefig( save_filepath )
	plt.close()

def seg_img_compare():

	img_names = [ 
		'12003'
		,'161062'
		,'100080'
		,'172032' 
	]
	
	FONT_SIZE = 20
	for algo_name, VAR_DICT in tqdm( algo_names, unit='Algos' ):
		for var_name in tqdm( VAR_DICT, unit='Vars', leave=False ):
			
			var_val_count = len( VAR_DICT[var_name] )
			f,ax = plt.subplots(len(img_names),var_val_count+2,figsize=[24,24])
			
			for r, img_name in tqdm( enumerate( img_names ), unit='Images', leave=False ):
				
				# SHOW ORIGINAL IMAGE
				img_filepath = get_img_filepath( img_name )
				img_array = np.array( Image.open( img_filepath ) )
				ax[r,0].imshow( img_array )
				ax[r,0].set_title( img_name,fontweight="bold", size=FONT_SIZE )
				ax[r,0].axis('off')

				# SHOW AN IDEAL SEGMENTATION
				seg_filepath = list( get_berkeley_seg_filepaths( img_name ) )[0]
				seg_array = load_segmentation( seg_filepath )
				segment_count = np.amax(seg_array)+1
				color_map = get_cmap('hsv',segment_count)
				ax[r,1].imshow( seg_array, cmap=color_map )
				ax[r,1].set_title( 'IDEAL SEG',fontweight="bold", size=FONT_SIZE )
				ax[r,1].axis('off')

				for c,var_val in tqdm( enumerate( VAR_DICT[var_name] ), unit='Var Vals', leave=False ):
					
					# LOAD SEG ARRAY FROM MEMORY
					var_val_seg_filepath = f"{ALGO_SEG_DIR}\\{algo_name}\\{var_name}\\{var_val}\\{img_name}.seg"
					seg_array = load_segmentation( var_val_seg_filepath )
					segment_count = np.amax(seg_array)+1
					color_map = get_cmap('hsv',segment_count)
					
					# DISPLAY SEG ARRAY
					ax[r,c+2].imshow( seg_array, cmap=color_map )
					ax[r,c+2].set_title( var_val,fontweight="bold", size=FONT_SIZE )
					ax[r,c+2].axis('off')

			plt.tight_layout()
			plt.subplots_adjust(hspace=0.2,wspace=0)
			save_filepath = f"{REPORT_DIR}\\images_{algo_name}_{var_name}.jpg"
			plt.savefig( save_filepath )
			plt.close()

def one_off_box_whisker_plot( summary_filepath, title ):
	with open( summary_filepath, 'r' ) as ofile:
		reader = csv.reader( ofile, delimiter=',' )
		next(reader,None)
		summary_data = [row[1:] for row in reader]
		summary_data = np.array( summary_data ).astype('float')
	fig=plt.figure(1,figsize=(9,6))
	ax=fig.add_subplot(111)
	bp=ax.boxplot(summary_data,patch_artist=True)
	plt.title( title, fontweight='bold', fontsize=12 )
	plt.ylabel( 'NPR (Normalized Rand Index)', fontweight='bold', fontsize=12 )
	plt.xlabel( '1', fontweight='bold', fontsize=12 )
	for field in bp:
		plt.setp(bp[field],color='green')
	plt.setp(bp['boxes'],facecolor='turquoise')
	for line in bp['medians']:
		x,y=line.get_xydata()[1]
		str_label = '%.3f' % y
		ax.annotate( str_label, xy=(x,y+0.01), fontweight='bold', fontsize=8 )
	plt.tight_layout()
	save_filepath = f"{REPORT_DIR}\\boxplot_{title}.jpg"
	plt.savefig( save_filepath )
	plt.close()

def box_whisker_plots():
	for algo_name, VAR_DICT in tqdm( algo_names, unit='Algos' ):
		for var_name in tqdm( VAR_DICT.keys(), unit='Vars', leave=False ):
			summary_filepath = ALGO_SEG_DIR + f'\\{algo_name}\\{var_name}\\summary.csv'
			with open( summary_filepath, 'r' ) as ofile:
				reader = csv.reader( ofile, delimiter=',' )
				next(reader,None)
				summary_data = [row[1:] for row in reader]
				summary_data = np.array( summary_data ).astype('float')
			fig=plt.figure(1,figsize=(9,6))
			ax=fig.add_subplot(111)
			bp=ax.boxplot(summary_data,patch_artist=True)
			ax.set_xticklabels(VAR_DICT[var_name])
			plt.title( f'NPR vs. {var_name}', fontweight='bold', fontsize=12 )
			plt.ylabel( 'NPR (Normalized Rand Index)', fontweight='bold', fontsize=12 )
			plt.xlabel( var_name, fontweight='bold', fontsize=12 )
			for field in bp:
				plt.setp(bp[field],color='green')
			plt.setp(bp['boxes'],facecolor='turquoise')
			for line in bp['medians']:
				x,y=line.get_xydata()[1]
				str_label = '%.3f' % y
				ax.annotate( str_label, xy=(x,y+0.01), fontweight='bold', fontsize=8 )
			plt.tight_layout()
			save_filepath = f"{REPORT_DIR}\\boxplot_{algo_name}_{var_name}.jpg"
			plt.savefig( save_filepath )
			plt.close()

# one_off_box_whisker_plot( r"C:\Users\conno\OneDrive\Desktop\Masters_2\algos_segmentations\fuzzy_cmeans\optimal\summary.csv", 'Optimal Fuzzy C-Means NPR' )
# seg_img_compare()
# box_whisker_plots()
# one_off_seg_img_compare( r"C:\Users\conno\OneDrive\Desktop\Masters_2\algos_segmentations\fuzzy_cmeans\optimal", 'OPTIMAL', 'fuzzy_cmeans_optimal', ['35070','42049'] )