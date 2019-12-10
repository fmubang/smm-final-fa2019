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

def get_new_old_counts(orig_df, tag, platforms=["youtube", "twitter"]):

	all_dfs = []

	for platform in platforms:

		df = orig_df.copy()
		df = df[df["platform"]==platform].copy()

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

		cols = ["platform","num_old_rootUsers", "num_new_rootUsers", "num_old_nodeUsers", "num_new_nodeUsers"]
		count_df = pd.DataFrame(data=data)
		count_df["platform"] = platform
		# count_df["data_type"] = tag
		count_df = count_df[cols]
		all_dfs.append(count_df)

	count_df = pd.concat(all_dfs)
	# print(count_df)

	return count_df

def get_new_old_counts(orig_df, tag, platforms=["youtube", "twitter"]):

	all_dfs = []

	for platform in platforms:

		df = orig_df.copy()
		df = df[df["platform"]==platform].copy()

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

			user_count_tag_list = ["num_old_rootUsers", "num_new_rootUsers","num_old_nodeUsers", "num_new_nodeUsers" ]
			user_count_list = [old_root_count, new_root_count, old_node_count, new_node_count ]
			data={"category":user_count_tag_list, "num_unique": user_count_list}

		# data={"num_old_rootUsers":[old_root_count],
		# 	"num_new_rootUsers":[new_root_count],
		# 	"num_new_nodeUsers":[new_node_count],
		# 	"num_old_nodeUsers":[old_node_count],
		# 	}

		# cols = ["platform","num_old_rootUsers", "num_new_rootUsers", "num_old_nodeUsers", "num_new_nodeUsers"]
		cols = ["platform","dataset","category", "num_unique"]
		count_df = pd.DataFrame(data=data)
		count_df["dataset"] = tag
		count_df["platform"] = platform
		
		# count_df["data_type"] = tag
		count_df = count_df[cols]
		all_dfs.append(count_df)

	count_df = pd.concat(all_dfs)

	cur_sum = count_df["num_unique"].sum()
	count_df["frequency"] = np.round(count_df["num_unique"]/cur_sum, 4)
	count_df = count_df.sort_values("frequency", ascending=False).reset_index(drop=True)
	return count_df



train_count_df = get_new_old_counts(train_df, "train")
# print(train_count_df)
# train_count_df = train_count_df.T
# train_count_df = train_count_df.rename(columns={0:"train"})
print(train_count_df)
# sys.exit(0)
print("\n\n")
test_count_df = get_new_old_counts(test_df, "test")
# test_count_df = test_count_df.T
# test_count_df = test_count_df.rename(columns={0:"test"})
print(test_count_df)

# count_df = pd.concat([train_count_df, test_count_df]).reset_index(drop=True)
# print("\n\n")
# print(count_df)

sys.exit(0)

output_dir = "Unique-Old-New-User-Counts/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_fp = output_dir + "train-unique-new-old-user-counts.csv"
train_count_df.to_csv(output_fp)
print(output_fp)

output_fp = output_dir + "test-unique-new-old-user-counts.csv"
test_count_df.to_csv(output_fp)
print(output_fp)




