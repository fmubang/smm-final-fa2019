import pandas as pd 
import numpy as np 
import os,sys

#train data
#input fp
train_fp = "/data/Fmubang/cp3-user-embed-exp/V4-EMBED-STUFF-11-30-FIXED-User-Embed-Materials/Vocab-Set-2018-04-01-to-2018-09-01/train-data-2018-09-01-to-2019-03-15.csv"
train_df = pd.read_csv(train_fp)
print(train_df)

#test data
test_fp = "/data/Fmubang/cp3-user-embed-exp/V4-EMBED-STUFF-11-30-FIXED-User-Embed-Materials/Vocab-Set-2018-04-01-to-2018-09-01/test-data-2019-03-15-to-2019-05-01.csv"
test_df = pd.read_csv(test_fp)
print(test_df)

#get tokens of interest
TOKENS_OF_INTERST = ["nodeUserID", "rootUserID", "informationID", "actionType"]

#return a df with:
#num old roots, new roots, old nodes, new nodes

def get_new_old_counts(df, tag):

	#get root counts
	root_users = list(df["rootUserID"].unique())
	old_root_count = 0
	new_root_count = 0

	for ru in root_users:
		if "new" in ru:
			new_root_count+=1
		else:
			old_root_count +=1

	#get node  counts
	node_users = list(df["nodeUserID"].unique())
	old_node_count = 0
	new_node_count = 0

	for nu in node_users:
		if "new" in nu:
			new_node_count+=1
		else:
			old_node_count+=1

	data={"num_old_rootUsers":[old_root_count],
		"num_new_rootUsers":[new_root_count],
		"num_new_nodeUsers":[new_node_count],
		"num_old_nodeUsers":[old_node_count],
		}

	cols = ["num_old_rootUsers", "num_new_rootUsers", "num_old_nodeUsers", "num_new_nodeUsers"]
	count_df = pd.DataFrame(data=data)
	# count_df["data_type"] = tag
	count_df = count_df[cols]
	return count_df

train_count_df = get_new_old_counts(train_df, "train")
train_count_df = train_count_df.T
train_count_df = train_count_df.rename(columns={0:"train"})
print(train_count_df)

test_count_df = get_new_old_counts(test_df, "test")
test_count_df = test_count_df.T
test_count_df = test_count_df.rename(columns={0:"test"})
print(test_count_df)

count_df = pd.concat([train_count_df, test_count_df], axis=1)
print("\n\n")
print(count_df)

output_dir = "Unique-Old-New-User-Counts/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_fp = output_dir + "unique-new-old-user-counts.csv"
count_df.to_csv(output_fp)
print(output_fp)