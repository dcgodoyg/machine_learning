---
title: "Machine Learning Course Project"
author: "by Daniel Godoy"
---

**This document contains the course project for the Machine Learning Course of the**
**Johns Hopkins' Data Science Specialization at Coursera.**

## 1. Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to
collect a large amount of data about personal activity relatively inexpensively.
These type of devices are part of the quantified self movement – a group of enthusiasts
who take measurements about themselves regularly to improve their health, to find
patterns in their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but they rarely
quantify how well they do it. In this project, the goal is to use data from
accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were
asked to perform barbell lifts correctly and incorrectly in 5 different ways. More
information is available from the website here: http://groupware.les.inf.puc-rio.br/har
(see the section on the Weight Lifting Exercise Dataset). 

### 1.1. Data
The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## 2. Assignment
In order to complete the assignment, the followung steps were taken.

* Load the data
* Clean the data
* Prepare and Perform a Random Forest Model
* Predict resuls of the testing dataset.

### 2.1. Load the Data
The files were downloaded from the links provided before. The data was loaded into
two different datasets: *trainingfile* and *testingfile*.




```r
trainingfile <- read.csv(trainingfileroute, header=TRUE,
                     na.strings=c("NA", ""))
testingfile <- read.csv(testingfileroute, header=TRUE,
                     na.strings=c("NA", ""))
```

### 2.2. Clean the Data
In order to prepare a good model, I removed data that may not contribute to the
ability to predict of the model. First, I removed variables that identified the
subject of study and the time the data was taken. Then, I removed variables
that contained more than 90% of NAs. Then, I removed data that does not vary
significantly. 

Note that the data cleaning process was done to the trainingfile and testinfile
dataset equally. Every process is explained below:

#### 2.2.1. Removing Identification and Information Variables
After a quick inspection of the datasets, I noticed that the first six variables
corresponded to identification variables or information variables.  Since these
variables may not add prediction capacitiy to a model, I removed them.


```r
trainingfile <- trainingfile[, -(1:6)]
testingfile <- testingfile[, -(1:6)]
```

#### 2.2.2. Removing Variables Composed Mainly by NAs
After a quick inspection of the datasets, I noticed that there were many variables
that were composed mainly by NAs (missing data). Since these variables may not
add prediction capacity to a model, I removed them.


```r
minimumrows <- nrow(trainingfile)*0.9
naspercolumn <- sapply(trainingfile, function(x) sum(is.na(x)))
trainingfile <- trainingfile[, naspercolumn < minimumrows]

minimumrows <- nrow(testingfile)*0.9
naspercolumn <- sapply(testingfile, function(x) sum(is.na(x)))
testingfile <- testingfile[, naspercolumn < minimumrows]
```

#### 2.2.3. Removing Low Varying Variables
After a quick inspection of the datasets, I noticed that there were vaiables that
did not vary much. Since these variables may not add prediction capacity to a model,
I removed them. For doing this, I removed variables which its
[Coefficient of Variation](https://en.wikipedia.org/wiki/Coefficient_of_variation).
was less than 1. 


```r
variation <- sapply(trainingfile, function(x) abs(sd(x)/mean(x)))
```

```
## Warning in mean.default(x): argument is not numeric or logical: returning
## NA
```

```r
trainingfile <- trainingfile[ , -variation[-length(variation)] < 1]

variation <- sapply(testingfile, function(x) abs(sd(x)/mean(x)))
testingfile <- testingfile[ , -variation[-length(variation)] < 1]
```


### 2.3. Prepare and Perform and Random Forest Model
To create a model that can predict the classe certain data belongs to, 
I used the Random Forest algorithm in the training dataset. The reasons I chose
the Random Forest algorithm are:

* The algorithm does not expect linear features on the variables
* The algorithm does not expect lienar relationships among variables
* The algorithm maages well large number of training samples
* The algorithm gives estimates of what variables are important in the classification
* The algorithm there does not need cross-validation as it is estimated internally,
during the execution.

In order to prepare the data, I created a training and testing data set out of
the *trainingfile*.

#### 2.3.1. Creating the Training and Testing Data Set

To create the Random Forest algorithm, the data on the trainingfile were split into
two data sets: training and testing.


```r
library(caret)
inTrain <- createDataPartition(trainingfile$classe, p=0.75, list=FALSE)
training <- trainingfile[inTrain, ]
testing <- trainingfile[-inTrain, ]
```

#### 2.3.2. Creating the Random Forest Algorithm

The random forest algorithm was crated using the randomForest package in R.


```r
library(randomForest)
set.seed(1234)
modelFit <- randomForest(classe~., data=training, importance=TRUE)
```

The results of the algorithm were:


```r
modelFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4184    0    0    0    1 0.0002389486
## B    4 2842    2    0    0 0.0021067416
## C    0    8 2559    0    0 0.0031164784
## D    0    0   10 2401    1 0.0045605307
## E    0    0    0    7 2699 0.0025868441
```

As the results show, every time we only randomly used 7 predictorts. Additionally,
The error rate is really small.

#### 2.3.3. Evaluation of the Altogrithm
To evaluate the algorithm, we will predict the outcome(classe) of the testing dataset
created in section 2.3.1 and evaluate the results using a Confusion Matrix.


```r
confusionMatrix(predict(modelFit, newdata=testing[, -ncol(testing)]),
                testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  949    2    0    0
##          C    0    0  853    4    0
##          D    0    0    0  800    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9988          
##                  95% CI : (0.9973, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9977   0.9950   1.0000
## Specificity            1.0000   0.9995   0.9990   1.0000   1.0000
## Pos Pred Value         1.0000   0.9979   0.9953   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   0.9995   0.9990   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1935   0.1739   0.1631   0.1837
## Detection Prevalence   0.2845   0.1939   0.1748   0.1631   0.1837
## Balanced Accuracy      1.0000   0.9997   0.9983   0.9975   1.0000
```

As the table above shows, the accuracy of the prediction model s pretty good. 
Additionally, the kappa measurement is also very good.


### 3. Predicting the Classe of the Testing File
I predicted the class of the observations that were on the testing file using
the algorithm built in section 2.3.2. 


```r
predictions <- predict(modelFit,newdata=testingfile)
predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

The data was used in the Submission of the course project. 

---
END

