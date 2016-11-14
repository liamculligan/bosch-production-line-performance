#Bosch Production Line Performance

#Train a stage 1 generaliser (gradient boosted decision tree) using stage 0 out-of-fold model predictions as inputs

#Authors: Tyrone Cragg & Liam Culligan
#Date: November 2016

#Load required packages
library(data.table)
library(corrplot)
library(xgboost)

#Load Ids and target variables
train = fread("train_numeric.csv", select = c("Id", "Response"))
test = fread("test_numeric.csv", select = "Id")

#Set required keys prior to joining
setkey(train, Id)
setkey(test, Id)

#Read in stage 0 model predictions to create stage 1 data

for (version in 1:6) {
  
  trainFilename = paste("Train-V", version, ".csv", sep="")
  testFilename = paste("Test-V", version, ".csv", sep="")
  
  trainImport = fread(trainFilename)
  testImport = fread(testFilename)
  
  setkey(trainImport, Id)
  setkey(testImport, Id)
  
  train = train[trainImport]
  test = test[testImport]
  
}

#Extract vector of Id and remove Id from the feature set
trainId = train[, Id]
train[, Id := NULL]

testId = test[, Id]
test[, Id := NULL]

#Extract target variable and remove from feature set
trainResponse = train[, Response]
train[, Response := NULL]

#Create a correlation plot of stage 1 data - stacked generalisation performs better when models are not highly correlated
M = cor(train)
M[upper.tri(M)] = 0
diag(M) = 0
corrplot(M, method = "square")

#Extract column names
names_train = names(train)

#Convert the feature set and target variable to an xgb.DMatrix - required as an input to xgboost
dtrain = xgb.DMatrix(data = data.matrix(train), label = trainResponse, missing = 9999999)

#Remove data.table
rm(train)
gc()

#Initialise the data frame Results - will be used to save the results of a grid search
Results = data.frame(eta=NA, max_depth = NA, subsample = NA, colsample_bytree = NA, MCC=NA, st_dev=NA, threshold = NA, n_rounds=NA)

#Fix the early stop round for boosting
early_stop_round = 10

#Set the parameters to be used within the grid search
eta_values = c(0.05, 0.1,  0.3)
max_depth_values = c(2, 3, 4)
subsample_values = c(0.95)
colsample_bytree_values = c(0.8)

#Create a function to calculate Matthews's Correlation Coefficient
#Loop through different threshold values to be used as an input to the function
#Significantly less computationally demanding than doing this in stage 0, as only 6 features used in stage 1

for (threshold in seq(0.2, 0.5, 0.05)) {
  mcc_eval = function(preds, dtrain) {
    labels = getinfo(dtrain, "label")
    positives = as.logical(labels) # label to boolean
    counter = sum(positives) # get the amount of positive labels
    tp = as.numeric(sum(preds[positives] >= threshold))
    fp = as.numeric(sum(preds[!positives] >= threshold))
    tn = as.numeric(length(labels) - counter - fp) # avoid computing he opposite
    fn = as.numeric(counter - tp) # avoid computing the opposite
    mcc = (tp*tn-fp*fn)/(sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return(list(metric="mcc", value=ifelse(is.na(mcc), -1, mcc)))
  }
  
  #Loop through all combinations of model parameters and mcc threshold
  
  for (eta in eta_values) {
    for (max_depth in max_depth_values) {
      for (subsample in subsample_values) {
        for (colsample_bytree in colsample_bytree_values) {
          
          #Set the model parameters and mcc threshold for the current iteration
          
          param = list(objective = "binary:logistic",
                       booster = "gbtree",
                       eval_metric = mcc_eval,
                       eta = eta,
                       max_depth = max_depth,
                       subsample = subsample,
                       colsample_bytree = colsample_bytree
          )
          
          #Train the model using 5-fold cross validation with the given parameters
          
          set.seed(44)
          XGBcv = xgb.cv(params = param,
                         data = dtrain,
                         nrounds = 10000,
                         verbose = T,
                         nfold = 5,
                         early.stop.round = early_stop_round,
                         maximize = T
          )
          
          #Save the number of boosting rounds for this set of parameters
          n_rounds = length(XGBcv$test.mcc.mean) - early_stop_round
          
          #Save the Matthew's Correlation Coefficient obtained using 5-fold cross validation for this set of parameters
          MCC = XGBcv$test.mcc.mean[n_rounds]
          
          #Save the standard deviation of the scoring metric for this set of parameters
          st_dev = XGBcv$test.mcc.std[n_rounds]
          
          #Save the set of parameters and model tuning results in the data frame Results
          Results = rbind(Results, c(eta, max_depth, subsample, colsample_bytree, MCC, st_dev, threshold, n_rounds))
          
          rm(XGBcv)
          gc(reset=T)
        }
      }
    }
  }
}

#Order Results in descending order according to the scoring metric
Results = na.omit(Results)
Results = Results[order(-Results$MCC),]

#Set the best threshold for MCC
threshold = Results$threshold[1]

#Redefine MCC function using best threshold
mcc_eval = function(preds, dtrain) {
  labels = getinfo(dtrain, "label")
  # positives = (labels == 1) # might be faster depending on your CPU (and how you ordered labels)
  positives = as.logical(labels) # label to boolean
  counter = sum(positives) # get the amount of positive labels
  tp = as.numeric(sum(preds[positives] >= threshold))
  fp = as.numeric(sum(preds[!positives] >= threshold))
  tn = as.numeric(length(labels) - counter - fp) # avoid computing he opposite
  fn = as.numeric(counter - tp) # avoid computing the opposite
  mcc = (tp*tn-fp*fn)/(sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
  return(list(metric="mcc", value=ifelse(is.na(mcc), -1, mcc)))
}

#Set the best tuning paramters obtained using the previous grid search
paramTuned = list(objective = "binary:logistic",
                  booster = "gbtree",
                  eval_metric = mcc_eval,
                  eta = Results$eta[1],
                  max_depth = Results$max_depth[1],
                  subsample = Results$subsample[1],
                  colsample_bytree = Results$colsample_bytree[1]
)

#Train the model on the full training set using the best parameters
set.seed(44)
XGB = xgboost(params = paramTuned, 
              data = dtrain,
              nrounds = Results$n_rounds[1],
              verbose = 1
)

#Compute feature importance matrix
Importance = xgb.importance(names_train, model = XGB)

#Plot important features
xgb.plot.importance(Importance[1:nrow(Importance),])

#Make Predictions on the Test Set
PredTest = predict(XGB, data.matrix(test), missing = 9999999)

#Create Submission File and Write CSV
PredTest = ifelse(PredTest > threshold, 1, 0)
Submission = data.frame(Id = testId, Response = PredTest)
write.csv(Submission, "XGB_Stage_1_test.csv", row.names = F)
