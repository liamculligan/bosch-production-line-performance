#Bosch Production Line Performance

#Preprocess Data
#Many columns are duplicated. Use digest to detect and filter duplicate columns
#Reduce RAM requirement by reading columns in batches

#Authors: Tyrone Cragg and Liam Culligan

#Date: October 2016

#Load required packages
library(data.table)
library(digest)

#Function for fread
read = function(file, colClasses, select) {
  
  fread(file,
        colClasses = colClasses,
        na.strings = "",
        showProgress = F,
        select = select)
}

#CATEGORICAL VARIABLES
#Setup read parameters and lists to store outputs
n_batch  = 5
col_idx  = c(1:2141)
col_bat  = cut(col_idx, n_batch, labels = c(1:n_batch))

all_features = vector("list", n_batch)
all_digests  = vector("list", n_batch)

#Loop through each batch
#Only the feature names and digests of each feature are stored.
for(i in seq_along(all_features)) {
  
  print(i)
  
  dt = read(file = "train_categorical.csv", colClasses = "character", select = col_idx[col_bat == i])
  
  all_features[[i]] = names(dt)
  all_digests[[i]]  = lapply(dt, digest)
  
  rm(dt)
  gc()
}

#Check summary of feature names and digests
#Appears to be over 1,900 duplicates, including empty columns
feature_summary = data.table(feature = unlist(all_features),
                             digest  = unlist(all_digests))

#For the second duplicated value onwards, sets duplicated to TRUE
feature_summary$duplicate = duplicated(feature_summary$digest)

cat("There are an estimated", sum(feature_summary$duplicate), "duplicated columns")

names_to_keep = feature_summary[duplicate == F, feature]

#Read in data without duplictaed columns
train_categorical = fread("train_categorical.csv", select = names_to_keep)
test_categorical = fread("test_categorical.csv", select = names_to_keep)

#NUMERIC VARIABLES
#Setup read parameters and lists to store outputs
n_batch  = 4
col_idx  = c(1:970)
col_bat  = cut(col_idx, n_batch, labels = c(1:n_batch))

all_features = vector("list", n_batch)
all_digests  = vector("list", n_batch)

#Loop through each batch
#Only the feature names and digests of each feature are stored.
for(i in seq_along(all_features)) {
  
  print(i)
  
  dt = read(file = "train_numeric.csv", colClasses = "numeric", select = col_idx[col_bat == i])
  
  all_features[[i]] = names(dt)
  all_digests[[i]]  = lapply(dt, digest)
  
  rm(dt)
  gc()
}

#Check summary of feature names and digests
#Appears to be over 1900 duplicates, including empty columns
feature_summary = data.table(feature = unlist(all_features),
                             digest  = unlist(all_digests))

#For the second duplicated value onwards, sets duplicated to TRUE
feature_summary$duplicate = duplicated(feature_summary$digest)

cat("There are an estimated", sum(feature_summary$duplicate), "duplicated columns")

#Keep all columns that are not duplicates
names_to_keep = feature_summary[duplicate == F, feature]

#Read in data without duplictaed columns -- still need to read in by batch to read RAM requirement
n_batch  = 4
col_idx  = c(1:length(names_to_keep))
col_bat  = cut(col_idx, n_batch, labels = c(1:n_batch))

#Loop through each batch
#Only the feature names and digests of each feature are stored.
for(i in 1:n_batch) {
  
  print(i)
  
  dt = read(file = "train_numeric.csv", colClasses = "numeric", select = names_to_keep[col_bat == i])
  
  if (i == 1) {
    train_numeric = dt
  } else {
    train_numeric = cbind(train_numeric, dt)
  }
  rm(dt)
  gc()
}

for(i in 1:n_batch) {
  
  print(i)
  
  dt = read(file = "test_numeric.csv", colClasses = "numeric", select = names_to_keep[col_bat == i])
  
  if (i == 1) {
    test_numeric = dt
  } else {
    test_numeric = cbind(test_numeric, dt)
  }
  rm(dt)
  gc()
}

#DATE VARIABLES
#Setup read parameters and lists to store outputs
n_batch  = 6
col_idx  = c(1:1157)
col_bat  = cut(col_idx, n_batch, labels = c(1:n_batch))

all_features = vector("list", n_batch)
all_digests  = vector("list", n_batch)

#Loop through each batch
#Only the feature names and digests of each feature are stored.
for(i in seq_along(all_features)) {
  
  print(i)
  
  dt = read(file = "train_date.csv", colClasses = "numeric", select = col_idx[col_bat == i])
  
  all_features[[i]] = names(dt)
  all_digests[[i]]  = lapply(dt, digest)
  
  rm(dt)
  gc()
}

#Check summary of feature names and digests
#Appears to be over 1,900 duplicates, including empty columns
feature_summary = data.table(feature = unlist(all_features),
                             digest  = unlist(all_digests))

#For the second duplicated value onwards, sets duplicated to TRUE
feature_summary$duplicate = duplicated(feature_summary$digest)

cat("There are an estimated", sum(feature_summary$duplicate), "duplicated columns")

#Keep all columns that are not duplicates
names_to_keep = feature_summary[duplicate == F, feature]

#Read in data without duplictaed columns -- still need to read in by batch to read RAM requirement
n_batch  = 6
col_idx  = c(1:length(names_to_keep))
col_bat  = cut(col_idx, n_batch, labels = c(1:n_batch))

#Loop through each batch
#Only the feature names and digests of each feature are stored.
for(i in 1:n_batch) {
  
  print(i)
  
  dt = read(file = "train_date.csv", colClasses = "numeric", select = names_to_keep[col_bat == i])
  
  if (i == 1) {
    train_date = dt
  } else {
    train_date = cbind(train_date, dt)
  }
  rm(dt)
  gc()
}

for(i in 1:n_batch) {
  
  print(i)
  
  dt = read(file = "test_date.csv", colClasses = "numeric", select = names_to_keep[col_bat == i])
  
  if (i == 1) {
    test_date = dt
  } else {
    test_date = cbind(test_date, dt)
  }
  rm(dt)
  gc()
}

#Save datasets as RData file
save(train_categorical, file = "PreProcTrainCategorical.rda", compress = T)
save(test_categorical, file = "PreProcTestCategorical.rda", compress = T)

save(train_numeric, file = "PreProcTrainNumeric.rda", compress = T)
save(test_numeric, file = "PreProcTestNumeric.rda", compress = T)

save(train_date, file = "PreProcTrainDate.rda", compress = T)
save(test_date, file = "PreProcTestDate.rda", compress = T)

#Randomly sample 100000 ovservations from training data
#This expedites the model training and validation process. Subsequently full training set can be used for training and validation.
set.seed(44)
samples = sample(nrow(train_numeric), 100000) #Random sample of 100000 row indeces
train_numeric_sample = train_numeric[samples] #Select these rows in the i argument of data.table
train_categorical_sample = train_categorical[samples]
train_date_sample = train_date[samples]

#Save sampled datasets as RData file
save(train_numeric_sample, train_categorical_sample, train_date_sample, file = "PreProcTrainSample.rda", compress = T)
