#Bosch Production Line Performance

#Obtain predictions using the trained gradient boosted decision tree

#Authors: Tyrone Cragg & Liam Culligan
#Date: November 2016

#Load required packages
library(data.table)
library(stringr)
library(ggplot2)
library(xgboost)

#Load Testing Data
load('PreProcTestNumeric.rda')
load('PreProcTestDate.rda')
load('PreProcTestCategorical.rda')

#Read in SortFeatures2
sort_features_2 = fread("SortFeatures2.csv")

#Load data relating to station path
load("TestStationPath.rda")

#Count the number of stations for each part (row)
test_station_path$num_stations = str_count(test_station_path$path, "|") 

#Remove uneccesary features from test_station_path
test_station_path[, ':=' (path_LOO_Unique = NULL,
                    path_LOO = NULL,
                    path = NULL,
                    count_test = NULL,
                    count_all= NULL)]

###
#Find the last station that each part goes to in the production line

#First obtain all feature names in train_date
test_date_features_last = test_date[, !"Id", with=F]

#Replace NAs with 0 -- otherwise NA returned
f_dowle3 = function(DT, missing_value) {
  for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j ,missing_value)
}

f_dowle3(test_date_features_last, 0)

#Manipulate a data.table into the required format
DT = data.table(value=unlist(test_date_features_last, use.names=FALSE), 
                colid = 1:nrow(test_date_features_last), rowid = rep(names(test_date_features_last), each=nrow(test_date_features_last)))

#Set required keys prior to joining
setkey(DT, colid, value)

#From DT, obtain the last line-station-feature that each part went to, that is the station with the greatest timestamp. Using the last
#column found as a tiebreaker.
last_station = DT[J(unique(colid), DT[J(unique(colid)), value, mult="last"]), rowid, mult="last"]

#Extract the last station from the last line-station-feature
last_station = gsub("L[0-9]+_", "", last_station)
last_station = gsub("_D[0-9]+", "", last_station)

#Find First Station - similar process to above

#First obtain all features names in test_date
test_date_features_first = test_date[, !"Id", with=F]

#Replace NAs with 9999999 -- otherwise NA returned
f_dowle3(test_date_features_first, 9999999)

#Manipulate a data.table into the required format
DT = data.table(value=unlist(test_date_features_first, use.names=FALSE), 
                colid = 1:nrow(test_date_features_first), rowid = rep(names(test_date_features_first), each=nrow(test_date_features_first)))

#Set required keys prior to joining
setkey(DT, colid, value)

#From DT, obtain the first line-station-feature that each part went to, that is the station with the greatest timestamp.
#Using the first column found as a tiebreaker
first_station = DT[J(unique(colid), DT[J(unique(colid)), value, mult="first"]), rowid, mult="first"]

#Extract the first station from the first line-station-feature
first_station = gsub("L[0-9]+_", "", first_station)
first_station = gsub("_D[0-9]+", "", first_station)

#Add both vectors to train_date
test_date[, ':=' (first_station = first_station,
                  last_station = last_station)]

#Remove uneccesary data
rm(first_station, last_station, DT, test_date_features_first, test_date_features_last)
gc()

###

#Create further features from train_numeric and train_categorical

#Load feature names of train_numeric and train_categorical
load("XGB3 Train features_numeric.rda")
features_numeric = head(features_numeric, -1)
load("XGB3 Train features_categorical.rda")

#Count of values by row in test_numeric
test_numeric[, numeric_count := rowSums(test_numeric[, (features_numeric), with = F], na.rm = T)]

#Count of Positive Values in test_numeric

test_numeric[, positive_numeric_count := rowSums(test_numeric[, (features_numeric), with = F] > 0, na.rm = T)]

#Count of Negative Values in test_numeric

test_numeric[, negative_numeric_count := rowSums(test_numeric[, (features_numeric), with = F] < 0, na.rm = T)]

#Count of zeros in test_numeric

test_numeric[, zero_numeric_count := rowSums(test_numeric[, (features_numeric), with = F] == 0, na.rm = T)]

#Count of NAs in test_numeric

test_numeric[, missing_numeric_count := rowSums(is.na(test_numeric[, (features_numeric), with = F]))]

#Repeat the same procedure but for each unique line and station

#Load unique line_station names 
load("XGB3 Train unique_lines.rda")

for (line in unique_lines) {
  var_name = paste0(line, "_numeric_count")
  test_numeric[, (var_name) := rowSums(test_numeric[, (grep(line , names(test_numeric))), with = F], na.rm = T)]
  var_name = paste0(line, "_positive_numeric_count")
  test_numeric[, (var_name) := rowSums(test_numeric[, (grep(line , names(test_numeric))), with = F] > 0, na.rm = T)]
  var_name = paste0(line, "_negative_numeric_count")
  test_numeric[, (var_name) := rowSums(test_numeric[, (grep(line , names(test_numeric))), with = F] < 0, na.rm = T)]
  var_name = paste0(line, "_zero_numeric_count")
  test_numeric[, (var_name) := rowSums(test_numeric[, (grep(line , names(test_numeric))), with = F] == 0, na.rm = T)]
  var_name = paste0(line, "_missing_numeric_count")
  test_numeric[, (var_name) := rowSums(is.na(test_numeric[, (grep(line , names(test_numeric))), with = F]))]
}

#Count of values by station only - too many stations to determine each type of count as above

#Load unique stations
load("XGB3 Train unique_stations.rda")

for (station in unique_stations) {
  var_name = paste0(station, "_numeric_count")
  test_numeric[, (var_name) := rowSums(test_numeric[, (grep(station , names(test_numeric))), with = F], na.rm = T)]
}

#Count by different factor levels - the same factor level (e.g. T1) can be present in multiple columns

#Load the unique values for all the categorical features
load("XGB3 Train levels.rda")

for (level in levels) {
  var_name = paste0(level, "_categorical_count")
  test_categorical[, (var_name) := rowSums(test_categorical[, (features_categorical), with = F] == level, na.rm = T)]
}

#Remove unnecessary vectors
rm(features_numeric, features_categorical)
gc()

#Set required keys prior to joining
setkey(test_numeric, Id)
setkey(test_categorical, Id)
setkey(test_date, Id)
setkey(sort_features_2, Id)
setkey(test_station_path, Id)

#Join data and remove unneccesary data.tables
test = test_date[test_numeric]
rm(test_numeric, test_date)
gc()
test = test_categorical[test]
rm(test_categorical)
gc()
test = sort_features_2[test]
rm(sort_features_2)
gc()
test = test_station_path[test]
rm(test_station_path)
gc()

#Extract Vectors
testId = test$Id

#L0_S3_F69 all NA - remove
test[, ':=' (L0_S3_F69 = NULL)]

#Replace NAs with 9999999
f_dowle3(test, 9999999)

#Return all column names containing character variables
load("XGB3 Train character_names.rda")

#From al character variables remove "T" and convert to numeric
test[, (character_names) := lapply(character_names, function(x) { as.integer(gsub("T", "", test[[x]])) })] #NAs represent ""

#Replace NAs introduced from factor encoding with 0 - these won't be handled as missing values but rather a unique factor level
f_dowle3(test, 0)

#Load the Training Model
load("XGB3.rda")

#Load the Best Threshold
load("XGB3_best_thresh.rda")

#Make Predictions on the Test Set
PredTest = predict(XGB, data.matrix(test), missing = 9999999)

#Create Ensemble File and Write CSV
Version = 3
Preds = data.frame(pred = PredTest, Id = testId)
Filename = paste("Test-V", Version, ".csv", sep="")
Colname1 = paste("V", Version, sep="")
colnames(Preds) = c(Colname1, "Id")
write.csv(Preds, Filename, row.names=F)

#Create Submission File and Write CSV
PredTest = ifelse(PredTest > quantile(PredTest, best_thresh), 1, 0)
Submission = data.frame(Id = testId, Response = PredTest)
write.csv(Submission, "XGB3_test.csv", row.names = F)


