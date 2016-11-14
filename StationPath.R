#Bosch Production Line Performance

#Determine the stations that each part (row) goes to.
#This results in a high cardinality catageorical variable which is then encoded using "leave-one-out encoding".
#That is, for each unique station combination, the mean Response is calculated for all training rows, although the current row is not 
#considered. Following this, the values obtained are slightly randomised to prevent overfitting.

#Authors: Tyrone Cragg and Liam Culligan

#Date: October 2016

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)

#Load necessary data
load("PreProcTrainCategorical.rda")
load("PreProcTestCategorical.rda")
load("PreProcTrainNumeric.rda")
load("PreProcTestNumeric.rda")

#Find unique stations in numeric training data from column names
#Get column names - exclude Id and Response
numeric_names = names(train_numeric)[!names(train_numeric) %in% c("Id", "Response")]
numeric_names = numeric_names[!numeric_names %in% c("Id", "Response")]
#Extract text between underscores - these are the station numbers
numeric_names = substr(numeric_names, regexpr('_', numeric_names)+2, 100)
numeric_names = substr(numeric_names, 0, regexpr('_', numeric_names)-1)
#Only keep unique stations
numeric_stations = unique(numeric_names)

#Find unique stations in categorical training data from column names
#Get column names - exclude Id and Response
categorical_names = names(train_categorical)[!names(train_categorical) %in% c("Id", "Response")]
#Extract text between underscores - these are the station numbers
categorical_names = substr(categorical_names, regexpr('_', categorical_names)+2, 100)
categorical_names = substr(categorical_names, 0, regexpr('_', categorical_names)-1)
#Only keep unique stations
categorical_stations = unique(categorical_names)

#Combine numeric and categorical stations, get unique stations, convert to numbers and order
stations = as.integer(unique(c(numeric_stations, categorical_stations)))
stations = stations[order(stations)]

#Extract number of rows
train_rows = nrow(train_numeric)
test_rows = nrow(test_numeric)

#Add empty Response to testing data for combining
test_numeric[, Response := NA]

#Combine training and testing data
data_numeric = rbind(train_numeric, test_numeric)
data_categorical = rbind(train_categorical, test_categorical)

#Create empty data frame for finding stations for each row
station_sums = data.frame(Id = data_categorical$Id)

#Loop through each station
for (station in stations) {
  
  #Create station name string
  station_name = paste0("_S", station, "_")
  
  #Locate columns belonging this station
  numeric_cols = grep(station_name, colnames(data_numeric))
  categorical_cols = grep(station_name, colnames(data_categorical))
  
  #Calculate number of these columns with non-missing values for each row - training
  if (length(numeric_cols) > 0) {
    numeric_row_sums = rowSums(!is.na(data_numeric[, numeric_cols, with=F]) &
                                 (data_numeric[, numeric_cols, with=F] != ""))
  } else {
    numeric_row_sums = 0
  }
  if (length(categorical_cols) > 0) {
    categorical_row_sums = rowSums(!is.na(data_categorical[, categorical_cols, with=F]) &
                                     (data_categorical[, categorical_cols, with=F] != ""))
  } else {
    categorical_row_sums = 0
  }
  
  #Calculate total number of visits to this station for each row and assign to station vaiable - training
  station_sums[[paste0("S", station)]] = numeric_row_sums + categorical_row_sums
  
}

station_sums = as.data.table(station_sums)

#Create path string - training
station_sums[, path := ""]

#Get station columns
station_cols = names(station_sums)
station_cols = station_cols[2:(length(station_cols)-1)]

#For each station, if station is present, add to path string for row
for (station_col in station_cols) {
  station_sums[, path := ifelse(station_sums[, station_col, with=F] > 0, paste0(path, station_col, "|"), path)]
}

#Count number of rows for each unique path
summary = station_sums %>%
  group_by(path) %>%
  summarise(count = n())

#Separate training and testing data
summary_train = head(summary, train_rows)
summary_test = tail(summary, test_rows)

#Set path as factor
station_sums[, path := as.factor(path)]

#Add Response
station_sums[, Response := data_numeric$Response]

#Separate training and testing station_sums
station_sums_train = head(station_sums, train_rows)
station_sums_test = tail(station_sums, test_rows)

#LOO encoding
LooEncoding = function(train, test, factorVec, targetName, idName) {
  #factorName,targetName and idName provided as strings

  data = data.table(id = c(train[[idName]], test[[idName]]))
  names(data) = idName
  
  setkeyv(data, idName)
  
  for (factorName in factorVec) { 
    
    LOO = train[, .(meanTarget = unlist(lapply(.SD, mean)), n = unlist(lapply(.SD, length))),
                .SDcols = targetName, by = factorName]
    
    setkeyv(LOO, factorName)
    setkeyv(train, factorName)
    
    trainTemp = LOO[train]
    
    #Add randomness
    set.seed(44)
    trainRand = runif(nrow(trainTemp), 0.95, 1.05)
    
    trainTemp[, meanTarget := (((meanTarget * n) - trainTemp[[targetName]])/(n-1)) * trainRand] 
    
    setkeyv(test, factorName)
    
    testTemp = LOO[test]
    
    dataTemp = data.table(id=c(trainTemp[[idName]], testTemp[[idName]]),
                          var=c(trainTemp$meanTarget, testTemp$meanTarget))
    names(dataTemp) = c(idName, factorName)
    
    setkeyv(dataTemp, idName)
    
    data = dataTemp[data]
    
  }
  
  return(data)
  #Returns data.frame of ids and the result of LOO Hot Encoding. Ready to merge with train and test by id.
  
}

#Apply LOO encoding function to data
PathLOO1 = LooEncoding(station_sums_train, station_sums_test, "path", "Response", "Id")

#Rename columns
names(PathLOO1) = c("Id", "path_LOO")

#Get training rows with unique paths
unique_path_train = subset(summary_train, count == 1)
unique_path_train_paths = unique_path_train$path

#Reassign above unique paths with a new, common name
station_sums_train[path %in% unique_path_train_paths, path := "Unique"]
station_sums_test[path %in% unique_path_train_paths, path := "Unique"]

#Reapply LOO encoding function to data
PathLOO2 = LooEncoding(station_sums_train, station_sums_test, "path", "Response", "Id")

#Rename columns
names(PathLOO2) = c("Id", "path_LOO_Unique")

#Counts for each path in train
count_train = station_sums_train %>%
  group_by(path) %>%
  summarise(count_train = n()) %>%
  as.data.table

#Counts for each path in test
count_test = station_sums_test %>%
  group_by(path) %>%
  summarise(count_test = n()) %>%
  as.data.table

#Counts for each path in full data
station_sums_test$Response = NA

#Combine training and testing station_sums
station_sums_all = rbind(station_sums_train, station_sums_test)

#Counts for each path in data
count_all = station_sums_all %>%
  group_by(path) %>%
  summarise(count_all = n()) %>%
  as.data.table

#Create data.tables for counts
train_counts = data.table(Id = station_sums_train$Id, path = station_sums_train$path)
test_counts = data.table(Id = station_sums_test$Id, path = station_sums_test$path)

#Create appropriate keys for joining
setkey(train_counts, path)
setkey(test_counts, path)
setkey(count_train, path)
setkey(count_test, path)
setkey(count_all, path)

#Combine counts in training and testing data
train_counts = count_train[train_counts]
train_counts = count_test[train_counts]
train_counts = count_all[train_counts]

test_counts = count_train[test_counts]
test_counts = count_test[test_counts]
test_counts = count_all[test_counts]

#Create appropriate keys for joining
setkey(train_counts, Id)
setkey(test_counts, Id)
setkey(PathLOO1, Id)
setkey(PathLOO2, Id)

#Join LOO data with counts
train_station_path = PathLOO1[train_counts]
train_station_path = PathLOO2[train_station_path]
names(train_station_path) = c("Id", "path_LOO_Unique", "path_LOO", "path", "count_all", "count_test", "count_train")

test_station_path = PathLOO1[test_counts]
test_station_path = PathLOO2[test_station_path]
names(test_station_path) = c("Id", "path_LOO_Unique", "path_LOO", "path", "count_all", "count_test", "count_train")

#Save RData
save(train_station_path, file = "TrainStationPath.rda", compress=T)
save(test_station_path, file = "TestStationPath.rda", compress=T)

