import pandas as pd 
import numpy as np 
import os,sys
from functools import *

def config_df_by_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df=df.set_index(time_col)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=False)
    return df

def get_token_counts(orig_df, TOKENS_OF_INTERST, tag, platforms=["youtube", "twitter"]):

    all_dfs = []

    for platform in platforms:

        df = orig_df.copy()
        df = df[df["platform"]==platform].reset_index(drop=True)
        #save counts here for df
        token_count_list = []
        for token in TOKENS_OF_INTERST:
            unique_tokens = df[token].nunique()
            token_count_list.append(unique_tokens)

        #make df
        data = {"token_type":TOKENS_OF_INTERST, "%s_count"%tag:token_count_list}
        new_df = pd.DataFrame(data=data)

        cols = ["platform","token_type", "%s_count"%tag]
        new_df["platform"] = platform
        new_df = new_df[cols]

        all_dfs.append(new_df)

    # new_df=reduce(lambda left,right: pd.merge(left,right,on=cols,how="inner"), all_dfs)
    new_df = pd.concat(all_dfs).reset_index(drop=True)
    print(new_df)
    # sys.exit(0)

    return new_df

output_dir = "Unique-Token-Counts/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#get stat counts for the training data
#get number of users, actions, etc AFTER CLEANING
#we need -> number of each token category

#get vocab data
pre_train_vocab_fp = "/data/Fmubang/cp3-user-embed-exp/V5-FIXED-PRE-VOCAB-EMBED-STUFF-11-30-FIXED-User-Embed-Materials/Vocab-Set-2018-04-01-to-2018-09-01/2018-04-01-to-2018-09-01-pre-train-whole-data.csv"
ptv_df = pd.read_csv(pre_train_vocab_fp)
print(ptv_df)

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



train_token_counts = get_token_counts(train_df, TOKENS_OF_INTERST, "train")
print(train_token_counts)

test_token_counts = get_token_counts(test_df, TOKENS_OF_INTERST, "test")
print(test_token_counts)

pre_train_token_counts = get_token_counts(ptv_df, TOKENS_OF_INTERST, "pre_train_vocab")
print(pre_train_token_counts)

token_counts = pd.merge(pre_train_token_counts, train_token_counts, on=["token_type", "platform"], how="inner")
token_counts = pd.merge(token_counts, test_token_counts, on=["token_type", "platform"], how="inner")

print("\n\n")
print(token_counts)

#save it
token_count_fp =output_dir + "Unique-Token-Counts.csv"
token_counts.to_csv(token_count_fp,index=False)
print(token_count_fp)

