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
from sklearn.linear_model import LinearRegression
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

#DEBUG PARAM
DEBUG = False
LIMIT_DIV_FACTOR = 30
MOD_NUM = 10

#minmax flag
SCALE_DATA_FLAGS = [True, False]

#log norm
LOG_NORM_FLAGS = [True, False]


if DEBUG == True:
	SCALE_DATA_FLAGS = [False]

#SAVE DFS HERE
DFS = []

IO_TUPLES = [
# (30,10),
# (30,5),
# (30,1),
# (20,10),
# (20,5),
# (20,1),
(10,10),
(10,5),
(10,1)
]

if DEBUG == True:
	IO_TUPLES = [(10,1)]

for SLIDING_WINDOW in [False]:

	if SLIDING_WINDOW == True:
		SW_TAG = "WITH_SLIDING_WINDOW_"
	else:
		SW_TAG = "WITHOUT_SLIDING_WINDOW_"

	for IO_TUPLE in IO_TUPLES:

		INPUT_EVENT_NUM = IO_TUPLE[0]
		OUTPUT_EVENT_NUM = IO_TUPLE[1]
		tag = "%s-to-%s"%(INPUT_EVENT_NUM, OUTPUT_EVENT_NUM)

		for SCALE_DATA in SCALE_DATA_FLAGS:

			for LOG_NORM_FLAG in LOG_NORM_FLAGS:

				#skip this for loop if this condition is met
				if (LOG_NORM_FLAG==True) and (SCALE_DATA==False):
					continue

				#num jobs for cos sim
				NUM_JOBS_FOR_COSINE = 10

				ALL_TAG = "all"
				TAGS = ["<nodeUserID>", "<rootUserID>","<informationID>","<actionType>", ALL_TAG]

				#perform knn with our new samples
				main_output_dir = "12-5-LINEAR-REGRESSION-NO-SLIDING-WINDOW/%s/SCALE-DATA-%s-LOG-NORM-%s-SLIDING-WINDOW-%s-DEBUG-%s/"%(tag,SCALE_DATA,LOG_NORM_FLAG,SLIDING_WINDOW,DEBUG)
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
					NEW_TRAIN_SIZE = int(x_train.shape[0]/LIMIT_DIV_FACTOR)
					x_train = x_train[:NEW_TRAIN_SIZE, :]
					y_train = y_train[:NEW_TRAIN_SIZE, :]

					NEW_TEST_SIZE = int(x_test.shape[0]/LIMIT_DIV_FACTOR)
					x_test = x_test[:NEW_TEST_SIZE, :]
					y_test = y_test[:NEW_TEST_SIZE, :]
					y_test_event_label_array = y_test_event_label_array[:NEW_TEST_SIZE, :]
					# y_test_event_label_array = y_test_event_label_array.flatten()

				y_test_event_label_array = y_test_event_label_array.flatten()
				orig_y_test_event_label_array = y_test_event_label_array.copy()

				#get shapes
				print("\ntrain shapes")
				print(x_train.shape)
				print(y_train.shape)
				print("\ntest shapes")
				print(x_test.shape)
				print(y_test.shape)

				# #NN OPTIONS
				# # KNN_ALGO_OPTIONS = ["auto", "ball_tree", "kd_tree", "brute"]
				# KNN_ALGO_OPTIONS = ["brute"]
				# # LEAF_SIZE_LIST = [10, 20, 30]
				# LEAF_SIZE_LIST = [10]
				# # DISTANCE_METRICS = ["euclidean", "chebyshev", "minkowski"]
				# # DISTANCE_METRICS = ["euclidean"]
				# # DISTANCE_METRICS = ["manhattan"]
				# DISTANCE_METRICS = ["euclidean", "manhattan", "chebyshev"]
				# NJOBS = 20
				# KNN_LIST = [3]
				# # WEIGHT_OPTION_LIST = ["uniform", "distance"]
				# WEIGHT_OPTION_LIST = ["distance"]
				# if DEBUG == True:
				# 	# KNN_ALGO_OPTIONS = KNN_ALGO_OPTIONS[:1]
				# 	KNN_ALGO_OPTIONS = ["brute"]
				# 	LEAF_SIZE_LIST = LEAF_SIZE_LIST[:1]
				# 	# DISTANCE_METRICS = DISTANCE_METRICS[:1]
				# 	KNN_LIST = KNN_LIST[:1]
				# 	WEIGHT_OPTION_LIST = ["distance"]
				# 	# DISTANCE_METRICS = ["chebyshev"]
				# 	# DISTANCE_METRICS = ["manhattan"]

				# 	KNN_LIST = KNN_LIST[:1]

				#make param dic

				#get train event to array dict
				efp = "/data/Fmubang/cp3-user-embed-exp/CP3-W2V-Lookup-Tables/SEQUENCE-SIZE-400-EMBED-SIZE-100-EPOCHS-10/TRAIN_TOKEN_TO_ARRAY_DICT.npy"
				EVENT_TO_ARRAY_DICT = np.load(efp, allow_pickle =True)

				#get train event labels
				train_event_label_fp = "/data/Fmubang/cp3-user-embed-exp/CP3-W2V-Lookup-Tables/SEQUENCE-SIZE-400-EMBED-SIZE-100-EPOCHS-10/train-full-event-labels.csv"
				train_event_label_df = pd.read_csv(train_event_label_fp)
				print(train_event_label_df["event_label"])
				train_event_labels = list(train_event_label_df["event_label"])

				#get keys in train dict
				ALL_TRAIN_EVENT_ARRAYS = []
				print("get ALL_TRAIN_EVENT_ARRAYS...")
				for label in train_event_labels:
					label_array = EVENT_TO_ARRAY_DICT.item().get(label)
					ALL_TRAIN_EVENT_ARRAYS.append(label_array)
				ALL_TRAIN_EVENT_ARRAYS = np.asarray(ALL_TRAIN_EVENT_ARRAYS)
				print(ALL_TRAIN_EVENT_ARRAYS[0])
				print("ALL_TRAIN_EVENT_ARRAYS shape")
				print(ALL_TRAIN_EVENT_ARRAYS.shape)
				# sys.exit(0)

				if SCALE_DATA == True:
					#need to scale data
					scaler = MinMaxScaler()
					scaler.fit(x_train)
					x_train = scaler.transform(x_train)
					x_test = scaler.transform(x_test)

					if (LOG_NORM_FLAG == True) and (SCALE_DATA==True):
						x_train = np.log1p(x_train)
						x_test = np.log1p(x_test)

				#save time info
				time_fp = main_output_dir + "time.txt"
				f = open(time_fp, "w")
				TIME_DICT = {}

				#make model
				model = LinearRegression()
				print(model)

				#train
				start = time.time()
				model.fit(x_train, y_train)
				end = time.time()
				total_time = end - start
				output_str = "Training time took %.2f seconds"%total_time
				f.write(output_str + "\n")
				TIME_DICT["train_time"] = output_str

				#make preds
				print("Predicting")
				start = time.time()
				y_regression_pred1 = model.predict(x_test)
				end = time.time()
				total_time = end - start
				output_str = "Testing time took %.2f seconds"%total_time
				f.write(output_str + "\n")
				TIME_DICT["test_time"] = output_str

				print("Shape of regression y_regression_pred1")
				print(y_regression_pred1.shape)

				print("Getting cosine sims...")

				#first reshape the preds
				print("Reshaping y preds")
				d1 = (y_regression_pred1.shape[0] * OUTPUT_EVENT_NUM)
				d2 = int(y_regression_pred1.shape[1] /OUTPUT_EVENT_NUM)
				y_regression_pred1 = y_regression_pred1.reshape((d1,d2))
				NUM_Y_PREDS = y_regression_pred1.shape[0]
				print("New y_regression_pred1 shape: %s" %str(y_regression_pred1.shape))

				#save event labels here
				y_regression_pred1_event_labels = []

				#parallel
				# get_closest_event_label(y_reg_pred, y_reg_pred_idx,ALL_TRAIN_EVENT_ARRAYS, train_event_labels)
				y_regression_pred1_event_labels = Parallel(n_jobs=NUM_JOBS_FOR_COSINE)(delayed(get_closest_event_label)(y_reg_pred, y_reg_pred_idx,ALL_TRAIN_EVENT_ARRAYS, train_event_labels, NUM_Y_PREDS) for y_reg_pred_idx,y_reg_pred  in enumerate(y_regression_pred1))

				##################### GET PRED LABELS #####################
				#split them up
				split_up_labels = []
				for label in y_regression_pred1_event_labels:
					# print(label.shape)
					# label = str(label)
					print(label)
					split_label = label.split("<with>")
					for e in split_label:
						split_up_labels.append(e)
				y_regression_pred1_event_labels = split_up_labels
				TOTAL_LABELS = len(split_up_labels)
				y_regression_pred1_event_labels = np.asarray(y_regression_pred1_event_labels)
				y_regression_pred1_event_labels = y_regression_pred1_event_labels.reshape((NUM_Y_PREDS, len(split_label)))


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
				y_regression_pred1_event_labels = y_regression_pred1_event_labels.reshape((TOTAL_PREDS//OUTPUT_EVENT_NUM, OUTPUT_EVENT_NUM))

				# y_regression_pred1_event_labels = np.asarray(y_regression_pred1_event_labels)
				print(y_regression_pred1_event_labels)
				print("y_regression_pred1_event_labels shape")
				print(y_regression_pred1_event_labels.shape)

				#let's save the predictions
				pred_data = {"prediction":list(y_regression_pred1_event_labels.flatten()), "actual":list(y_test_event_label_array.flatten())}

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

						X_TEST_SIZE = INPUT_EVENT_NUM 
						Y_TEST_SIZE = OUTPUT_EVENT_NUM 
						# y_test_event_label_array = y_test_event_label_array.flatten()
						# y_regression_pred1_event_labels = y_regression_pred1_event_labels.flatten()
						df = evaluate_prediction_without_builtin_tags(EVAL_STYLE,X_TEST_SIZE,Y_TEST_SIZE,y_test_event_label_array,y_regression_pred1_event_labels, FILTER_BY_TAG=TAG)

						df["scaled_data"] = SCALE_DATA
						df["used_log_norm"] = LOG_NORM_FLAG
						df["sliding_window_tag"] = SW_TAG
						final_cols = list(df)

						#add info
						df=df[final_cols]
						# df = df.sort_values("accuracy", ascending)
						print(df)

						DFS.append(df)

	#combine dfs
	df = pd.concat(DFS)
	df = df.reset_index(drop=True)
	df = df.sort_values("accuracy", ascending=False)

	# #save full df
	# output_fp = main_output_dir + "All-Sequence-Prediction-Evaluations.csv"
	# df.to_csv(output_fp, index=False)
	# print(output_fp)



	GROUPBY_COLS = ["tag", "eval_style","input_sequence_size","output_sequence_size", "scaled_data"] 
	DROP_COLS = ["sequence_num"]
	df["accuracy"] = df.groupby(GROUPBY_COLS)["accuracy"].transform("mean")
	df["num_correctly_predicted"] = df.groupby(GROUPBY_COLS)["num_correctly_predicted"].transform("sum")
	df["total_predictions"] = df.groupby(GROUPBY_COLS)["total_predictions"].transform("sum")
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








