"""
Bosch Production Line Performance

Find start time and end time for each part (row). Subsequently sort by various
columns and combinations of columns to find each part's interactions with
previous and next parts

Authors: Tyrone Cragg & Liam Culligan
Date: October 2016
"""

#Import required packages
import pandas as pd
import numpy as np

#Specify columns to create new features with
feature_columns = ['L3_S32_F3850', 'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859', 'L3_S33_F3865',
'L3_S33_F3867', 'L3_S33_F3869', 'L3_S33_F3871','L3_S33_F3873', 'L3_S34_F3876', 'L3_S34_F3878',
'L3_S34_F3880', 'L3_S34_F3882', 'L1_S24_F1723', 'L1_S24_F1846', 'L3_S38_F3952', 'L3_S30_F3494',
'L2_S27_F3129', 'L3_S30_F3754','L2_S26_F3069', 'L3_S30_F3744']

#Read in Id and feature columns from numeric data
train = pd.read_csv("train_numeric.csv", usecols=feature_columns+['Id'])
test = pd.read_csv("test_numeric.csv", usecols=feature_columns+['Id'])

#Initialise start and end time
train["StartTime"] = -1
test["StartTime"] = -1

train["EndTime"] = -1
test["EndTime"] = -1

#Get number of training rows
ntrain = train.shape[0]

#Get number of testing rows
ntest = test.shape[0]

#Set maximum number of rows in training or testing data so that we know when to break out of the loop
max_nrows = max(ntrain, ntest)

#Set number of rows to read in at a time
chunksize = 50000

#Locate earliest and latest time for each row - credit: Kaggle user Faron
nrows = 0

#Read in training and testing data in chunks of rows
for train_date, test_date in zip(pd.read_csv("train_date.csv", chunksize=chunksize),
                  pd.read_csv("test_date.csv", chunksize=chunksize)):
    
    #Look at all columns except Id
    feats = np.setdiff1d(train_date.columns, ['Id'])
    
    #Find minimum time for each row
    start_time_train_date = train_date[feats].min(axis=1).values
    start_time_test_date = test_date[feats].min(axis=1).values
    
    #Find maximum time for each row
    end_time_train_date = train_date[feats].max(axis=1).values
    end_time_test_date = test_date[feats].max(axis=1).values
    
    #Set minimum time as StartTime for corresponding Id in training and testing sets
    train.loc[train.Id.isin(train_date.Id), 'StartTime'] = start_time_train_date
    test.loc[test.Id.isin(test_date.Id), 'StartTime'] = start_time_test_date
    
    #Set maximum time as EndTime for corresponding Id in training and testing sets
    train.loc[train.Id.isin(train_date.Id), 'EndTime'] = end_time_train_date
    test.loc[test.Id.isin(test_date.Id), 'EndTime'] = end_time_test_date
    
    #Iterate nrows
    nrows += chunksize
    
    #Break if all rows have been read in
    if nrows >= max_nrows:
        break

#Combine training and testing sets and reset indices
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)

#Get total time for each row
train_test["TotalTime"] = train_test["EndTime"] - train_test["StartTime"]

#Define columns and combinations of columns to sort by
sorts = ['index', ['StartTime', 'Id'], ['StartTime', 'EndTime'], ['EndTime', 'Id'], ['TotalTime', 'Id'], 'Id']

#Extract features from next and previous columns by different sorting
for sort in sorts:
  
  #Set variable name prefix to combination of sorting column names
  name = ''.join(sort)
  
  #Sort
  train_test = train_test.sort_values(by=sort, ascending=True)
  
  #Loop through each feature column
  for feature_column in feature_columns:
    
    #Previous and next rows' feature values
    train_test[name+'Prev'+feature_column] = train_test[feature_column].shift(1).fillna(9999999)
    train_test[name+'Next'+feature_column] = train_test[feature_column].shift(-1).fillna(9999999)
    
    #Difference between this row's feature value and previous and next rows' feature values
    train_test['IndexPrev'+feature_column+'Diff'] = train_test[feature_column].diff().fillna(9999999)
    train_test['IndexNext'+feature_column+'Diff'] = train_test[feature_column].iloc[::-1].diff().fillna(9999999)

#Drop original columns
train_test = train_test.drop(['StartTime', 'EndTime', 'TotalTime',
'L3_S32_F3850', 'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859',
'L3_S33_F3865', 'L3_S33_F3867', 'L3_S33_F3869', 'L3_S33_F3871','L3_S33_F3873', 'L3_S34_F3876',
'L3_S34_F3878', 'L3_S34_F3880', 'L3_S34_F3882', 'L1_S24_F1723', 'L1_S24_F1846', 'L3_S38_F3952',
'L3_S30_F3494', 'L2_S27_F3129', 'L3_S30_F3754', 'L2_S26_F3069', 'L3_S30_F3744'], axis=1)

#Reorder indices for saving
train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

#Write CSV
train_test.to_csv("SortFeatures3.csv", index=False)


