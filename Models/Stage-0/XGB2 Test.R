#Bosch Production Line Performance

#Obtain predictions using the trained gradient boosted decision tree

#Authors: Tyrone Cragg & Liam Culligan
#Date: November 2016

#Load required packages
library(data.table)
library(ggplot2)
library(xgboost)

#Load Testing Data
load('PreProcTestNumeric.rda')
load('PreProcTestDate.rda')
load('PreProcTestCategorical.rda')

#Read in SortFeatures1
sort_features_1 = fread("SortFeatures1.csv")

#Set required keys prior to joining
setkey(test_numeric, Id)
setkey(test_categorical, Id)
setkey(test_date, Id)
setkey(sort_features_1, Id)

#Join data and remove unneccesary data.tables
test = test_date[test_numeric]
rm(test_numeric, test_date)
gc()
test = test_categorical[test]
rm(test_categorical)
gc()
test = sort_features_1[test]
rm(sort_features_1)
gc()

#Extract Vectors
testId = test$Id

#L0_S3_F69 all NA - remove
test[, ':=' (L0_S3_F69 = NULL)]

#Replace NAs with 9999999
f_dowle3 = function(DT, missing_value) {
  for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j ,missing_value)
}

f_dowle3(test, 9999999)

#Return all column names containing character variables
load("XGB2 Train character_names.rda")

#From al character variables remove "T" and convert to numeric
test[, (character_names) := lapply(character_names, function(x) { as.integer(gsub("T", "", test[[x]])) })] #NAs represent ""

#Replace NAs introduced from factor encoding with 0 - these won't be handled as missing values but rather a unique factor level
f_dowle3(test, 0)

#Load the Training Model
load("XGB2.rda")

#Load the Best Threshold
load("XGB2_best_thresh.rda")

#Make Predictions on the Test Set
PredTest = predict(XGB, data.matrix(test), missing = 9999999)

#Create Ensemble File and Write CSV
Version = 2
Preds = data.frame(pred = PredTest, Id = testId)
Filename = paste("Test-V", Version, ".csv", sep="")
Colname1 = paste("V", Version, sep="")
colnames(Preds) = c(Colname1, "Id")
write.csv(Preds, Filename, row.names=F)

#Create Submission File and Write CSV
PredTest = ifelse(PredTest > quantile(PredTest, best_thresh), 1, 0)
Submission = data.frame(Id = testId, Response = PredTest)
write.csv(Submission, "XGB2_test.csv", row.names = F)


