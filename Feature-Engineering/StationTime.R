#Bosch Production Line Performance

#For each station-feature, order the data by the time. For each row then obtain the previous and next value of Response. Also calculate
#the difference between that row's time and the previous and next rows' times.

#Authors: Tyrone Cragg and Liam Culligan

#Date: October 2016

#Load required packages
library(data.table)

#Only Require Id and Response from train_numeric
train_numeric = fread("train_numeric.csv", select = c("Id", "Response"))
test_numeric = fread("train_numeric.csv", select = c("Id"))

#Need Reponse column for test
test_numeric[, Response := NA]

#Load Training and Testing Dates
load('PreProcTrainDate.rda')
load('PreProcTestDate.rda')

#Set the necessary keys prior to joining
setkey(train_numeric, Id)
setkey(train_date, Id)
setkey(test_numeric, Id)
setkey(test_date, Id)

#Join the data
train_date_station = train_date[train_numeric]
test_date_station = test_date[test_numeric]

#Remove Unecessary Data
rm(train_date, train_numeric, test_date, test_numeric)
gc()

#Row bind train and test
date_station = rbind(train_date_station, test_date_station)

#Remove Unecessary Data
rm(train_date_station, test_date_station)
gc()

#Create vectors of new feature names
col_names = names(date_station)[!names(date_station) %in% c("Id", "Response")]
new_col_names_prev = paste("prev_diff", col_names, sep = "_")
new_col_names_following = paste("next_diff", col_names, sep = "_")
new_col_names_prev_response = paste("prev_response", col_names, sep = "_")
new_col_names_following_response = paste("next_response", col_names, sep = "_")

#Loop through each column in col_names. For each iteration, order the data_table by the current value in col_names.
for (i in 1:length(col_names)) {
  
  #For the current column in col_names, calculate the difference in time between the current row and the previous row
  date_station = date_station[order(get(col_names[i]), Id)][, (new_col_names_prev[i]) := lapply(.SD, function(x) c(NA, diff(x))),
                                                            .SDcols = col_names[i]]
  
  #For the current column in col_names, calculate the previous Response
  date_station[, (new_col_names_prev_response[i]) := shift(Response, 1L, type = "lag")]
  
  #For the current column in col_names, calculate the difference in time between the current row and the next row
  date_station = date_station[order(-get(col_names[i]), -Id)][, (new_col_names_following[i]) := lapply(.SD, function(x) c(NA, diff(x))),
                                                              .SDcols = col_names[i]]
  
  #For the current column in col_names, calculate the next Response
  date_station[, (new_col_names_following_response[i]) := shift(Response, 1L, type = "lead")]
  
  #Remove the current column in col_names from the data.table
  date_station[, (col_names[i]) := NULL]
  
  gc()
  cat(i/length(col_names) * 100, "% complete \n")
}

#Remove the target variable from date_station
date_station[, Response := NULL]

#Save date_station as RData file
save(date_station, file = "DateStation.rda", compress = T)

