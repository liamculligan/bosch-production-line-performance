#Bosch Production Line Performance

#Train a gradient boosted decision tree using 5-fold cross validation

#Authors: Tyrone Cragg & Liam Culligan
#Date: November 2016

#Load required packages
library(data.table)
library(ggplot2)
library(xgboost)

#Load Training Data
load('PreProcTrainNumeric.rda')
load('PreProcTrainDate.rda')
load('PreProcTrainCategorical.rda')

#Read in SortFeatures1
sort_features_1 = fread("SortFeatures1.csv")

#Set required keys prior to joining
setkey(train_numeric, Id)
setkey(train_categorical, Id)
setkey(train_date, Id)
setkey(sort_features_1, Id)

#Join data and remove unneccesary data.tables
train = train_date[train_numeric]
rm(train_numeric, train_date)
gc()
train = train_categorical[train]
rm(train_categorical)
gc()
train = sort_features_1[train]
rm(sort_features_1)
gc()

#Extract Vectors
trainId = train$Id
trainResponse = train$Response

#Remove the target variable from the feature set
train[, ':=' (Response = NULL)]

#L0_S3_F69 all NA - remove
train[, ':=' (L0_S3_F69 = NULL)]

#Replace NAs with 9999999
f_dowle3 = function(DT, missing_value) {
  for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j ,missing_value)
}

f_dowle3(train, 9999999)

#Return all column names containing character variables
character_cols = sapply(train, class) == "character"
character_names = names(train)[character_cols]

#Save character_names - necessary so that the same features can be used on the test data
save(character_names, file = "XGB2 Train character_names.rda")

#From all character variables remove "T" and convert to numeric
#One-hot-encoding of these features would require significantly more memory and result in a much larger feature set
#A tree-based model can make suitable splits on a factor encoded as an integer
train[, (character_names) := lapply(character_names, function(x) { as.integer(gsub("T", "", train[[x]])) })] #NAs represent ""

#Replace NAs introduced from factor encoding with 0 - these won't be handled as missing values but rather a unique factor level
f_dowle3(train, 0)

#Save the names of the feature set in order to assess their importance in the trained model
train_names = names(train)
save(train_names, file = "XGB2 Train train_names.rda")

#Convert the feature set and target variable to an xgb.DMatrix - required as an input to xgboost
dtrain = xgb.DMatrix(data = data.matrix(train), label = trainResponse, missing = 9999999)

#Remove data.table
rm(train)
gc()

#The model is trained using AUC as the evaluation metric. By doing this, and thereafter looping through threshold values
#to be applied using Matthews's Correlation Coefficient, as opposed to looping through these thresholds during model training,
#computational time is significantly reduced.

#Initialise the data frame Results - will be used to save the results of a grid search
Results = data.frame(eta=NA, max_depth = NA, subsample = NA, colsample_bytree = NA, AUC=NA, st_dev=NA, n_rounds=NA)

#Initialise the data frame OOSPreds - will be used to save the out-of-fold model predictions - needed for a stacked generalisation
OOSPreds = data.frame(pred=NA, Id=NA, eta=NA, max_depth=NA, subsample=NA, colsample_bytree=NA)

#Save the version number for the stacked generalisation
Version = 2

#Fix the early stop round for boosting
early_stop_round = 10

#Set the parameters to be used within the grid search - only best set of parameters found via tuning is retained here
eta_values = c(0.1)
max_depth_values = c(10)
subsample_values = c(0.95)
colsample_bytree_values = c(0.5)

#Loop through all combinations of the parameters to be used in the above grid
for (eta in eta_values) {
  for (max_depth in max_depth_values) {
    for (subsample in subsample_values) {
      for (colsample_bytree in colsample_bytree_values) {
        
        #Set the model parameters for the current iteration
        
        param = list(objective = "binary:logistic",
                     booster = "gbtree",
                     eval_metric = "auc",
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
                       maximize = T,
                       prediction = T
        )
        
        #Save the number of boosting rounds for this set of parameters
        n_rounds = length(XGBcv$dt$test.auc.mean) - early_stop_round
        
        #Save the AUC score obtained using 5-fold cross validation for this set of parameters
        AUC = XGBcv$dt$test.auc.mean[n_rounds]
        
        #Save the standard deviation of the scoring metric for this set of parameters
        st_dev = XGBcv$dt$test.auc.std[n_rounds]
        
        #Save the set of parameters and model tuning results in the data frame Results
        Results = rbind(Results, c(eta, max_depth, subsample, colsample_bytree, AUC, st_dev, n_rounds))
        
        #Add out-of-fold predictions to OOSPreds for this set of parameters
        PredIter = as.data.frame(XGBcv$pred)
        colnames(PredIter) = "pred"
        
        PredIter$Id = trainId
        
        PredIter$eta = eta
        PredIter$max_depth = max_depth
        PredIter$subsample = subsample
        PredIter$colsample_bytree = colsample_bytree
        
        OOSPreds = rbind(OOSPreds, PredIter)
        
        rm(XGBcv)
        gc()
      }
    }
  }
}

#Order Results in descending order according to the scoring metric
Results = na.omit(Results)
Results = Results[order(-Results$AUC),]

#Select only the out-of-fold predictions for the best set of parameters found above
OOSPreds = na.omit(OOSPreds)
OOSPreds = subset(OOSPreds, OOSPreds$eta == Results$eta[1] & OOSPreds$max_depth == Results$max_depth[1] &
                    OOSPreds$subsample == Results$subsample[1] &
                    OOSPreds$colsample_bytree == Results$colsample_bytree[1])

#Only require Id and the predicted probability as an input to the stacked generalisation
OOSPreds = OOSPreds[, c(1, 2)]

#Save the out-of-fold predicted probabilites to be used in the stacked generalisation
Filename = paste("Train-V", Version, ".csv", sep="")
Colname1 = paste("V", Version, sep="")

colnames(OOSPreds) = c(Colname1, "Id")

write.csv(OOSPreds, Filename, row.names=F)

#Calculate Matthew's Correlation Coefficient at various thresholds

mc = function(actual, predicted) {
  
  tp = as.numeric(sum(actual == 1 & predicted == 1))
  tn = as.numeric(sum(actual == 0 & predicted == 0))
  fp = as.numeric(sum(actual == 0 & predicted == 1))
  fn = as.numeric(sum(actual == 1 & predicted == 0))
  
  numer = (tp * tn) - (fp * fn)
  denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ^ 0.5
  
  numer / denom
}

matt = data.table(thresh = seq(0.97, 0.9999, by = 0.0001))

matt$score = sapply(matt$thresh, FUN =
                      function(x) mc(trainResponse, (OOSPreds$pred > quantile(OOSPreds$pred, x)) * 1))

matt = matt[order(-matt$score)]

#Create Plot to demonstrate threshold selection
ggplot(matt, aes(x = thresh, y = score)) + 
  geom_line()

#Select the threshold value that maximises the evaluation metric
best_thresh = matt$thresh[1]

#Save the best threshold
save(best_thresh, file = "XGB2_best_thresh.rda")

#Set the best tuning paramters obtained using the previous grid search
paramTuned = list(objective     = "binary:logistic",
                  booster            = "gbtree",
                  eval_metric        = "auc",
                  eta                = Results$eta[1],
                  max_depth          = Results$max_depth[1],
                  subsample          = Results$subsample[1],
                  colsample_bytree   = Results$colsample_bytree[1]
)

#Train the model on the full training set using the best parameters
set.seed(44)
XGB = xgboost(params              = paramTuned, 
              data                = dtrain,
              nrounds             = Results$n_rounds[1],
              verbose             = 1
)

#Save the Model
save(XGB, file = "XGB2.rda")

#Compute feature importance matrix
Importance = xgb.importance(train_names, model = XGB)

#Plot 10 most important features
xgb.plot.importance(Importance[1:10,])

