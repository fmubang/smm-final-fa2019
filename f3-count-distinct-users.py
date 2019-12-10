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

def count_distinct_users(orig_df, tag, platforms=["youtube", "twitter"]):

	print(orig_df)
	# sys.exit(0)

	all_dfs = []


	count_category_list = ["distinct_nodeUser", "distinct_rootUser",
		"nodeUser_cluster","rootUser_cluster"]

	for platform in platforms:



		df = orig_df.copy()
		df = df[df["platform"]==platform].copy()

				#count stuff
		original_root_count = 0
		changed_name_root_count = 0
		original_node_count = 0
		changed_name_node_count = 0

		original_nodes = list(df["original_nodeUserID"])
		original_roots = list(df["original_rootUserID"])
		nodes = list(df["nodeUserID"])
		roots = list(df["rootUserID"])

		distinct_node_set = set()
		distinct_root_set = set()
		clustered_node_set = set()
		clustered_root_set = set()

		


		for i in range(len(original_nodes)):

			if roots[i] == original_roots[i]:
				original_root_count+=1
				distinct_root_set.add(roots[i])
			else:
				changed_name_root_count+=1
				clustered_root_set.add(roots[i])

			if nodes[i] == original_nodes[i]:
				original_node_count+=1
				distinct_node_set.add(roots[i])
			else:
				changed_name_node_count+=1
				clustered_node_set.add(roots[i])

		count_list = [original_node_count, original_root_count, changed_name_node_count, changed_name_root_count]
		unique_user_count_list = [len(distinct_node_set), len(distinct_root_set), len(clustered_node_set), len(clustered_root_set)]

		data = {"category":count_category_list, "activity_count" : count_list, "unique_user_count":unique_user_count_list}
		count_df = pd.DataFrame(data=data)

		count_df["dataset"] = tag
		cols = ["platform","dataset", "category", "activity_count", "unique_user_count"]
		count_df["platform"] = platform

		all_dfs.append(count_df)

	count_df = pd.concat(all_dfs)
	count_df = count_df[cols]

	cur_sum = count_df["activity_count"].sum()
	count_df["activity_frequency"] = count_df["activity_count"]/cur_sum
	count_df = count_df.sort_values("activity_frequency", ascending=False).reset_index(drop=True)

	cur_sum = count_df["unique_user_count"].sum()
	count_df["unique_user_frequency"] = count_df["unique_user_count"]/cur_sum


	return count_df


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
# pd.option_context('display.max_rows', None, 'display.max_columns', None)

train_count_df = count_distinct_users(train_df, "train")
test_count_df = count_distinct_users(test_df, "test")

print(train_count_df)
print("\n\n")
print(test_count_df)
