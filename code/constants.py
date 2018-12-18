
import os

RANDOM_SEED = 0 

CODE_WORKING_DIR = os.path.dirname( os.path.realpath( __file__ ) )
WORKING_DIR = os.path.dirname( CODE_WORKING_DIR )

REPORT_DIR = WORKING_DIR + r'\report'

BERKELEY_IMG_DIR = WORKING_DIR + r"\berkeley_300_images"
BERKELEY_SEG_DIR = WORKING_DIR + r"\berkeley_300_segmentations"

ALGO_SEG_DIR = WORKING_DIR + r"\algos_segmentations"
FUZZY_CMEANS_DIR = ALGO_SEG_DIR + r"\fuzzy_cmeans"
KMEANS_DIR = ALGO_SEG_DIR + r"\kmeans"

FUZZY_CMEANS_OPTIMAL_SEG_DIR = FUZZY_CMEANS_DIR + r'\optimal'
KMEANS_OPTIMAL_SEG_DIR = KMEANS_DIR + r'\optimal'

DEFAULT_COLOR_SPACE='RGB'
DEFAULT_MAX_ITER=8
DEFAULT_N_CLUSTERS=8