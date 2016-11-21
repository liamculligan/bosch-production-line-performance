"""
Bosch Production Line Performance

Find start time and end time for each part (row). Subsequently sort by various
columns and combinations of columns to find each part's interactions with varying numbers of
previous and next parts

Authors: Tyrone Cragg & Liam Culligan
Date: October 2016
"""

#Import required packages
import pandas as pd
import numpy as np

#Only read in these columns
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
sorts = [['StartTime', 'Id'], ['StartTime', 'EndTime'], ['EndTime', 'Id'], ['TotalTime', 'Id'], 'Id']

#Define windows of number of rows to look backwards and forwards
windows = [10, 100, 1000, 10000]

#Extract features from next and previous columns by different sorting
for sort in sorts:
  
  #Set variable name prefix to combination of sorting column names
  name = ''.join(sort)
  
  #Sort
  train_test = train_test.sort_values(by=sort, ascending=True)
  
  #Loop through each specified window
  for window in windows:
    
    #Average of previous and next x rows' Responses
    train_test[name+'PrevResponse'+str(window)] = pd.rolling_mean(train_test['Response'], window=window, min_periods=1).shift(1)
    train_test[name+'NextResponse'+str(window)] = pd.rolling_mean(train_test['Response'][::-1], window=window, min_periods=1).shift(1)
    
    #Set first and last x values to NA (not enough rows)
    train_test[name+'PrevResponse'+str(window)][0:window] = np.nan
    train_test[name+'NextResponse'+str(window)][(len(train_test.index)-window):len(train_test.index)] = np.nan

#Drop original columns
train_test = train_test.drop(['StartTime', 'EndTime', 'TotalTime', 'Response'], axis=1)

#Reorder indices for saving
train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

train_test = train_test.fillna(9999999)

#Write CSV
train_test.to_csv("SortFeatures4.csv", index=False)

