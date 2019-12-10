import gensim 
import logging
import pandas as pd 
import os,sys
from scipy.spatial import distance
import time
import numpy as np 
from s0_w2v_pred_functions import *
from gensim.models import Word2Vec
import re
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cosDist
from joblib import Parallel,delayed
from sklearn.preprocessing import MinMaxScaler
import pickle

def get_closest_event_label(y_reg_pred, y_reg_pred_idx,ALL_TRAIN_EVENT_ARRAYS, train_event_labels,NUM_Y_PREDS):

	MOD_NUM = 10

	if y_reg_pred_idx%MOD_NUM == 0:
		print("Working on event label %d of %d" %((y_reg_pred_idx+1), NUM_Y_PREDS))

	NUM_CANDIDATES = ALL_TRAIN_EVENT_ARRAYS.shape[0]
	max_sim = -2
	max_sim_event_label =""
	for i, cvector in enumerate(ALL_TRAIN_EVENT_ARRAYS):
		# temp_sim = cosine_similarity(cvector, y_reg_pred)
		temp_sim = 1 - cosDist(cvector, y_reg_pred)
		if temp_sim > max_sim:
			max_sim = temp_sim
			max_sim_event_label = train_event_labels[i]

	if y_reg_pred_idx%MOD_NUM == 0:
		print("Got event label for y pred %d of %d" %((y_reg_pred_idx+1), NUM_Y_PREDS))

	return max_sim_event_label

def make_pseudo_prediction(x_test,Y_TEST_SIZE):
	return x_test[:,-Y_TEST_SIZE:]


#new baseline code!

IO_TUPLES = [
(30,10),
(30,5),
(30,1),
(20,10),
(20,5),
(20,1),
(10,10),
(10,5),
(10,1)
]

DEBUG=False
LIMIT_DIV_FACTOR = 30
MOD_NUM = 10
if DEBUG == True:
	IO_TUPLES = [(10,5)]


#tags for eval
ALL_TAG = "all"
TAGS = ["<nodeUserID>", "<rootUserID>","<informationID>","<actionType>", ALL_TAG]

#num jobs for cos sim
NUM_JOBS_FOR_COSINE = 20


for SLIDING_WINDOW in [True]:

	if SLIDING_WINDOW == True:
		SW_TAG = "WITH_SLIDING_WINDOW_"
	else:
		SW_TAG = "WITHOUT_SLIDING_WINDOW_"
	for IO_TUPLE in IO_TUPLES:

		#SAVE DFS HERE
		DFS = []
		
		#make our tag
		INPUT_EVENT_NUM = IO_TUPLE[0]
		OUTPUT_EVENT_NUM = IO_TUPLE[1]
		X_TEST_SIZE = IO_TUPLE[0]
		Y_TEST_SIZE = IO_TUPLE[1]
		tag = "%s-to-%s"%(INPUT_EVENT_NUM, OUTPUT_EVENT_NUM)
		

		#MAKE OUTPUT DIR
		main_output_dir = "ALL-BASELINES-/DEBUG-%s-SLIDING-WINDOW-%s/IO-%s/"%(DEBUG,SLIDING_WINDOW,tag)
		if not os.path.exists(main_output_dir):
			os.makedirs(main_output_dir)

			#load train and test data
		# main_input_dir = "/data/Fmubang/cp3-user-embed-exp/Samples-from-Lookup-Tables/%s/SEQUENCE-SIZE-400-EMBED-SIZE-100-EPOCHS-10/"%(tag)
		# main_input_dir = "/data/Fmubang/cp3-user-embed-exp/Samples-from-Lookup-Tables/%s/SEQUENCE-SIZE-400-EMBED-SIZE-100-EPOCHS-10/"%(tag)
		main_input_dir = "/data/Fmubang/cp3-user-embed-exp/BIG-SLIDING-WINDOW-DATA12-4-Better-Samples-from-Lookup-Tables/%s/SEQUENCE-SIZE-400-EMBED-SIZE-100-EPOCHS-10/"%(tag)

		#train data
		print("Loading training data...")
		# train_fp = main_input_dir + "WITH_SLIDING_WINDOW_TRAIN_LUT_XY_DICT.p"
		train_fp = main_input_dir + "%sTRAIN_LUT_XY_DICT.p"%SW_TAG
		# TRAIN_LUT_XY_DICT = np.load(train_fp, allow_pickle=True)
		pickle_in = open(train_fp,"rb")
		TRAIN_LUT_XY_DICT = pickle.load(pickle_in)
		print("Got train data!")

		print("\nLoading test data...")
		#test data
		test_fp = main_input_dir + "%sTEST_LUT_XY_DICT.p"%SW_TAG
		# TEST_LUT_XY_DICT = np.load(test_fp, allow_pickle=True)
		pickle_in = open(test_fp,"rb")
		TEST_LUT_XY_DICT = pickle.load(pickle_in)
		print("Test data!")

		#get train and test
		x_train = TRAIN_LUT_XY_DICT["x"]
		y_train = TRAIN_LUT_XY_DICT["y"]
		x_test = TEST_LUT_XY_DICT["x"]
		y_test = TEST_LUT_XY_DICT["y"]

		#for test data we need the event label array
		print("Getting test data label arrays")
		x_test_event_label_array = TEST_LUT_XY_DICT["x_event_labels"]
		y_test_event_label_array = TEST_LUT_XY_DICT["y_event_labels"]
		
		print("y_test_event_label_array shape")
		print(y_test_event_label_array.shape)
		
		# sys.exit(0)

		#if debug is true, use less samples
		if DEBUG==True:
			NEW_TEST_SIZE = int(x_test_event_label_array.shape[0]/LIMIT_DIV_FACTOR)
			x_test_event_label_array = x_test_event_label_array[:NEW_TEST_SIZE, :]
			y_test_event_label_array = y_test_event_label_array[:NEW_TEST_SIZE, :]
			# y_test_event_label_array = y_test_event_label_array.flatten()

		y_test_event_label_array = y_test_event_label_array.flatten()
		orig_y_test_event_label_array = y_test_event_label_array.copy()

		#get shapes
		print("\ntest shapes")
		print(x_test_event_label_array.shape)
		print(y_test_event_label_array.shape)

		#make pseudo pred
		y_pred_event_label_array = make_pseudo_prediction(x_test_event_label_array,Y_TEST_SIZE)
		print("Shape of y_pred_event_label_array")
		print(y_pred_event_label_array.shape)
		NUM_Y_PREDS = y_pred_event_label_array.shape[0]
		print("NUM_Y_PREDS")
		print(NUM_Y_PREDS)
		

		y_pred_event_label_array = y_pred_event_label_array.flatten()
		y_test_event_label_array = y_test_event_label_array.flatten()

		##################### GET PRED LABELS #####################
		#split them up
		split_up_labels = []
		for label in y_pred_event_label_array:
			# print(label.shape)
			label = str(label)
			# print(label)
			# sys.exit(0)
			split_label = label.split("<with>")
			for e in split_label:
				split_up_labels.append(e)
		y_pred_event_label_array = split_up_labels
		TOTAL_LABELS = len(split_up_labels)
		y_pred_event_label_array = np.asarray(y_pred_event_label_array)
		y_pred_event_label_array = y_pred_event_label_array.reshape((NUM_Y_PREDS, len(split_label) * OUTPUT_EVENT_NUM))


		##################### GET TEST LABELS #####################
		y_test_event_label_array = orig_y_test_event_label_array.copy()
		#split them up
		split_up_labels = []
		for label in y_test_event_label_array:
			# label = str(label)
			# print(label)
			split_label = label.split("<with>")
			for e in split_label:
				split_up_labels.append(e)
		y_test_event_label_array = split_up_labels
		TOTAL_LABELS = len(split_up_labels)
		y_test_event_label_array = np.asarray(y_test_event_label_array)
		TOTAL_PREDS = y_test_event_label_array.shape[0]

		y_test_event_label_array = y_test_event_label_array.reshape((TOTAL_PREDS//OUTPUT_EVENT_NUM, OUTPUT_EVENT_NUM ))
		y_pred_event_label_array = y_pred_event_label_array.reshape((TOTAL_PREDS//OUTPUT_EVENT_NUM, OUTPUT_EVENT_NUM))

		# y_pred_event_label_array = np.asarray(y_pred_event_label_array)
		print(y_pred_event_label_array)
		print("y_pred_event_label_array shape")
		print(y_pred_event_label_array.shape)

		#let's save the predictions
		pred_data = {"prediction":list(y_pred_event_label_array.flatten()), "actual":list(y_test_event_label_array.flatten())}

		#make pred df 
		pred_df = pd.DataFrame(data = pred_data)
		print(pred_df)

		# sys.exit(0)

		#save data
		output_fp =main_output_dir + "predictions-vs-ground-truth.csv"
		pred_df.to_csv(output_fp, index=False)
		print(output_fp)

		for TAG in TAGS:
			for EVAL_STYLE in ["bag_style", "exact_order_style"]:
				df = evaluate_prediction_without_builtin_tags(EVAL_STYLE,X_TEST_SIZE,Y_TEST_SIZE,y_test_event_label_array,y_pred_event_label_array, FILTER_BY_TAG=TAG)
				print(df)
				DFS.append(df)

		#combine dfs
		df = pd.concat(DFS)
		df = df.reset_index(drop=True)
		df = df.sort_values("accuracy", ascending=False)

		# #save full df
		# output_fp = output_dir + "All-Sequence-Prediction-Evaluations.csv"
		# df.to_csv(output_fp, index=False)
		# print(output_fp)

		GROUPBY_COLS = ["tag", "eval_style","input_sequence_size","output_sequence_size"]
		DROP_COLS = ["sequence_num"]
		df["accuracy"] = df.groupby(GROUPBY_COLS)["accuracy"].transform("mean")
		df["num_correctly_predicted"] = df.groupby(GROUPBY_COLS)["num_correctly_predicted"].transform("sum")
		df["total_predictions"] = df.groupby(GROUPBY_COLS)["total_predictions"].transform("sum")
		df["sliding_window_tag"] = SW_TAG
		df = df.drop(DROP_COLS,axis=1).drop_duplicates()
		df = df.sort_values("accuracy", ascending=False)
		df = df.reset_index(drop=True)
		print(df)

		#save
		#save full df
		output_fp = main_output_dir + "Summarized-Sequence-Prediction-Evaluations.csv"
		df.to_csv(output_fp, index=False)
		print(output_fp)
		print("Done")

		# sys.exit(0)
