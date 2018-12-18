from seg_loop import *
from constants import *
from algo_kmeans import K_MEANS_VAR_DICT
from algo_fuzzy_cmeans import FUZZY_CMEANS_VAR_DICT
import fire, traceback


def algo_looper( algo_name, var_name ):
	while True: # keep trying until we finish
		try:
			if algo_name == 'kmeans':
				val_range = K_MEANS_VAR_DICT[var_name]
			elif algo_name == 'fuzzy_cmeans':
				val_range = FUZZY_CMEANS_VAR_DICT[var_name]
			seg_looper( algo_name, var_name, val_range )
			break
		except Exception as err:
			traceback.print_exc()

if __name__ == '__main__':
	fire.Fire()