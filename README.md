# Bosch Production Line Performance
[Kaggle - Reduce manufacturing failures] (https://www.kaggle.com/c/bosch-production-line-performance)

## Introduction
The goal of this competition was to predict internal failures based on thousands of measurements and tests made for each component along the assembly line, using one of the largest datasets hosted on Kaggle to date. <br>
Each part was labelled as either passing quality control (Response = 0) or failing quality control (Response = 1) and model predictions were evaluated using [Matthew's Correlation Coefficient] (https://www.kaggle.com/c/bosch-production-line-performance/details/evaluation).

## Team Meambers
The team, Arrested Development, consisted of [Tyrone Cragg] (https://github.com/tyronecragg) and [Liam Culligan] (https://github.com/liamculligan).

## Solution Architecture
![Solution Architecture](https://github.com/liam-culligan/bosch-production-line-performance/blob/master/Solution%20Architecture.jpg?raw=true "Solution Architecture")

## Performance
The solution obtained a rank of [39th out of 1391 teams] (https://www.kaggle.com/c/bosch-production-line-performance/leaderboard/private). with a private leaderboard score of 0.48726. <br> The 5-fold cross validation Matthew's Correlation Coefficient was 0.47767, with a standard deviation of 0.00698.

## Execution
1. Create a working directory for the project <br>
2. [Download the data from Kaggle] (https://www.kaggle.com/c/bosch-production-line-performance/data) and place in the working directory.
3. Run `PreProcess.R`
4. Run feature engineering scripts: <br>
4.1 `SortFeatures1.py` <br>
4.2 `SortFeatures2.py` <br>
4.3 `SortFeatures3.py` <br>
4.4 `SortFeatures4.py` <br>
4.5 `StationPath.R` <br>
4.6 `StationTime.R` <br>
4.7 `DateRolling.R` <br>
5. Run the Stage 0 model scripts for the stacked generalisation: <br>
5.1 `XGB1 Train.R` and `XGB1 Test.R` <br>
5.2 `XGB2 Train.R` and `XGB2 Test.R` <br>
5.3 `XGB3 Train.R` and `XGB3 Test.R` <br>
5.4 `XGB4 Train.R` and `XGB4 Test.R` <br>
5.5 `XGB5 Train.R` and `XGB5 Test.R` <br>
5.6 `XGB6 Train.R` and `XGB6 Test.R` <br>
6. Run the Stage 1 model script, `XGB Stage 1.R`

## Requirements
* R 3+
* Python 3+
