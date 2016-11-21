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

#Read in Id and Response from numeric data
train = pd.read_csv("train_numeric.csv", usecols=['Id', 'Response'])
test = pd.read_csv("test_numeric.csv", usecols=['Id'])

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
  
  #Difference between this row's Id and previous and next rows' Ids
  train_test[name+'PrevIdDiff'] = train_test['Id'].diff().fillna(9999999).astype(int)
  train_test[name+'NextIdDiff'] = train_test['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
  
  #Difference between this row's StartTime and previous and next rows' StartTimes
  train_test[name+'PrevStartTimeDiff'] = train_test['StartTime'].diff().fillna(9999999).astype(int)
  train_test[name+'NextStartTimeDiff'] = train_test['StartTime'].iloc[::-1].diff().fillna(9999999).astype(int)
  
  #Difference between this row's EndTime and previous and next rows' EndTimes
  train_test[name+'PrevEndTimeDiff'] = train_test['EndTime'].diff().fillna(9999999).astype(int)
  train_test[name+'NextEndTimeDiff'] = train_test['EndTime'].iloc[::-1].diff().fillna(9999999).astype(int)
  
  #Difference between this row's TotalTime and previous and next rows' TotalTimes
  train_test[name+'PrevTotalTimeDiff'] = train_test['TotalTime'].diff().fillna(9999999).astype(int)
  train_test[name+'NextTotalTimeDiff'] = train_test['TotalTime'].iloc[::-1].diff().fillna(9999999).astype(int)
  
  #Previous and next rows' Ids
  train_test[name+'PrevId'] = train_test['Id'].shift(1).fillna(9999999).astype(int)
  train_test[name+'NextId'] = train_test['Id'].shift(-1).fillna(9999999).astype(int)
  
  #Previous and next rows' StartTimes
  train_test[name+'PrevStartTime'] = train_test['StartTime'].shift(1).fillna(9999999).astype(int)
  train_test[name+'NextStartTime'] = train_test['StartTime'].shift(-1).fillna(9999999).astype(int)
  
  #Previous and next rows' EndTimes
  train_test[name+'PrevEndTime'] = train_test['EndTime'].shift(1).fillna(9999999).astype(int)
  train_test[name+'NextEndTime'] = train_test['EndTime'].shift(-1).fillna(9999999).astype(int)
  
  #Previous and next rows' TotalTimes
  train_test[name+'PrevTotalTime'] = train_test['TotalTime'].shift(1).fillna(9999999).astype(int)
  train_test[name+'NextTotalTime'] = train_test['TotalTime'].shift(-1).fillna(9999999).astype(int)
  
  #Don't add for index sorting -- result would only present in training set, not testing set
  if sort != 'index':
    #Previous and next rows' Responses
    train_test['PrevResponse'] = train_test['Response'].shift(1).fillna(9999999).astype(int)
    train_test['NextResponse'] = train_test['Response'].shift(-1).fillna(9999999).astype(int)

#Drop original column
train_test = train_test.drop(['Response'], axis=1)

#Reorder indices for saving
train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

#Write CSV
train_test.to_csv("SortFeatures2.csv", index=False)

