#Bosch Production Line Performance

#Sort by each date feature and extract previous and next Responses for each row by different sorting

#Authors: Tyrone Cragg and Liam Culligan

#Date: October 2016

#Load required packages
library(data.table)
library(caret)

#Load necessary data
load("PreProcTrainDate.rda")

#Find near zero variance features

#Get column names - exclude Id
feature_names = names(train_date)[!names(train_date) %in% "Id"]

#Find near-zero variance columns
nzv_cols = nearZeroVar(train_date[, feature_names, with = F], freqCut = 99/1, names = T)

#Remove near-zero variance columns
cols_keep = setdiff(feature_names, nzv_cols)

#Add Id to cols_keep
cols_keep = c("Id", cols_keep)

#Keep only non-near-zero variance columns
train_date = train_date[, cols_keep, with=F]
#Read in only non-near-zero variance columns from test_date
test_date = fread("test_date.csv", select = cols_keep)

#Read in Id and Response from numeric data
train_numeric = fread("train_numeric.csv", select = c("Id", "Response"))
test_numeric = fread("test_numeric.csv", select = c("Id"))

test_numeric[, Response := NA]

data_date = rbind(train_date, test_date)
data_numeric = rbind(train_numeric, test_numeric)

#Create appropriate keys for joining
setkey(data_date, Id)
setkey(data_numeric, Id)

#Join numeric and date data
date_rolling = data_date[data_numeric]

#Get feature names
feature_names = names(date_rolling)[!names(date_rolling) %in% c("Id", "Response")]

#Loop through feature names
for (feature_name in feature_names) {
  
  #Set new column names for this feature
  feature_name_prev = paste(feature_name, "PrevResponse", sep="_")
  feature_name_next = paste(feature_name, "NextResponse", sep="_")
  
  #Sort by current feature
  setkeyv(date_rolling, feature_name)
  
  #Find rows where this feature is not missing
  rows = as.vector(!is.na(date_rolling[,feature_name, with=F]))
  
  #Get previous and next Response by this sorting for non-missing rows
  date_rolling[rows, feature_name_prev := shift(Response, 1L, type="lag"), with=F]
  date_rolling[rows, feature_name_next := shift(Response, 1L, type="lead"), with=F]
  
}

#Remove unnecessary columns
date_rolling[, feature_names := NULL, with=F]
date_rolling[, Response := NULL]

#Save RData
save(date_rolling, file = "DateRolling.rda", compress=T)

