# Kaggle - Titanic Survival Prediction

#### By Marcus Williamson - 02/09/15

Please see Kaggle Competition [here](https://www.kaggle.com/c/titanic)

>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
>
>One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
>
>In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.


### Importing and Cleaning Data

We begin by importing our libraries that we may need to use during the analysis and model creation, we will be comparing a wide variety of models:
```r
library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
library(NeuralNetTools)
library(rpart) 
library(rpart.plot)
library(kernlab)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)
library(nnet)
library(e1071)
library(MASS)
library(Matrix)
library(lme4)
library(arm)
library(survival)
library(splines)
library(gbm)
library(plyr)
library(klaR)
library(ipred)
```
We also set our seed for **reproduciblity of results** and number of cores available for processing with: 
```r
set.seed(8484) 
registerDoMC(cores = 8) # for parallel processing
```

Next we import our data from the source links into our R environment, a saved CSV copy can be found in the Data folder within the repository.
```r
rawdata = read.csv("train.csv", header = TRUE)
```

Taking an initial look at our data:
```r
dim(rawdata)
view(rawdata)
```
```
[1] 891  12
```

Under further visual inspection, it reveals variables in the raw data import that are not useful in prediction and are difficult to work with in a Machine Learning problem, so we proceed to clean the datasets:
```r
surv <- rawdata$Survived # keeping our survival variables aside
id <- rawdata$PassengerId # keeping passenger id

names <- grepl("Name|Ticket|Cabin|PassengerId", colnames(rawdata)) # getting all non predictive variables column names

rawdata = rawdata[,!names] # removing those columns

## futher cleansing of data incompatible with ML problems (na values)
data = rawdata[complete.cases(rawdata),]

data$Survived <- factor(data$Survived) # setting as factor
```
---

### Data Partitioning 

We split our training data into training and validation sets as we seek to implement [Cross Validation]( https://en.wikipedia.org/wiki/Cross-validation_(statistics) ) in this model. We choose a 70:30 split for our sets to ensure we have a suitable amount of data in each, and to reduce the variance in the parameter estimates. This is also roughly in line with the split of 60% Training 20% Validation (scaling up to give a 75:25 split).
```r
inTrain <- createDataPartition(y=data$Survived,p=0.70, list=FALSE)

training_part <- data[inTrain,] # 70% of data for training
testing_part <- data[-inTrain,] # 30% of data for later validation
```
---

##Preparing to compare multiple models

We create a vector to capture the different performances of the models, and set the Cross Validation settings with [Principle Component Analysis]() repeated 10 times
```r
# create an empty numeric vector to calculate out of sample error against
outOfSampleError <- numeric()

# add some parameters for train control
TC <- trainControl(method = "cv", number = 12, returnData=FALSE, returnResamp="none", savePredictions=FALSE, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
```

### Creating the Models

We build a wide variety of models, credit to [Nick Lust](https://github.com/nicklusk) for original script:

* [Bayesian GLM]()
* [Generalized Boosted Regression]()
* [K Nearest Neighbor]()
* [Naive Bayes]()
* [Neural Net]()
* [Random Forest]()
* [Recursive Partitioning and Regression Trees]()
* [Support Vector Machines Linear]()
* [Support Vector Machines Radial]()
* [Bagged Classification and Regression Trees]()

```r
# train, predict, calculate accuracy and out of sample error

#Bayesian
bayes <- train(Survived ~ ., method="bayesglm", data=training_part, trControl= TC)
bayesglmPrediction <- predict(bayes, testing_part)
bayesglmAccuracy <- sum(bayesglmPrediction == testing_part$Survived) / length(bayesglmPrediction)
bayesglmOutOfSampleError <- c(outOfSampleError, 1-bayesglmAccuracy)

gbm <- train(Survived ~ ., method="gbm", data=training_part, trControl= TC)
gbmPrediction <- predict(gbm, testing_part)
gbmAccuracy <- sum(gbmPrediction == testing_part$Survived) / length(gbmPrediction)
gbmOutOfSampleError <- c(outOfSampleError, 1-gbmAccuracy)

knn <- train(Survived ~ ., method="knn", data=training_part, trControl= TC)
knnPrediction <- predict(knn, testing_part)
knnAccuracy <- sum(knnPrediction == testing_part$Survived) / length(knnPrediction)
knnOutOfSampleError <- c(outOfSampleError, 1-knnAccuracy)

nb <- train(Survived ~ ., method="nb", data=training_part, trControl= TC)
nbPrediction <- predict(nb, testing_part)
nbAccuracy <- sum(nbPrediction == testing_part$Survived) / length(nbPrediction)
nbOutOfSampleError <- c(outOfSampleError, 1-nbAccuracy)

nnet <- train(Survived ~ ., method="nnet", data=training_part, trControl= TC)
nnetPrediction <- predict(nnet, testing_part)
nnetAccuracy <- sum(nnetPrediction == testing_part$Survived) / length(nnetPrediction)
nnetOutOfSampleError <- c(outOfSampleError, 1-nnetAccuracy)

rf <- train(Survived ~ ., method="rf", data=training_part, trControl= TC)
rfPrediction <- predict(rf, testing_part)
rfAccuracy <- sum(rfPrediction == testing_part$Survived) / length(rfPrediction)
rfOutOfSampleError <- c(outOfSampleError, 1-rfAccuracy)

rpart <- train(Survived ~ ., method="rpart", data=training_part, trControl= TC)
rpartPrediction <- predict(rpart, testing_part)
rpartAccuracy <- sum(rpartPrediction == testing_part$Survived) / length(rpartPrediction)
rpartOutOfSampleError <- c(outOfSampleError, 1-rpartAccuracy)

svml <- train(Survived ~ ., method="svmLinear", data=training_part, trControl= TC)
svmlPrediction <- predict(svml, testing_part)
svmlAccuracy <- sum(svmlPrediction == testing_part$Survived) / length(svmlPrediction)
svmlOutOfSampleError <- c(outOfSampleError, 1-svmlAccuracy)

svmr <- train(Survived ~ ., method="svmRadial", data=training_part, trControl= TC)
svmrPrediction <- predict(svmr, testing_part)
svmrAccuracy <- sum(svmrPrediction == testing_part$Survived) / length(svmrPrediction)
svmrOutOfSampleError <- c(outOfSampleError, 1-svmrAccuracy)

treebag <- train(Survived ~ ., method="treebag", data=training_part, trControl= TC)
treebagPrediction <- predict(treebag, testing_part)
treebagAccuracy <- sum(treebagPrediction == testing_part$Survived) / length(treebagPrediction)
treebagOutOfSampleError <- c(outOfSampleError, 1-treebagAccuracy)

#Results
trainMethods <- c("Bayesian GLM", "Generalized Boosted Regression", "K Nearest Neighbor", "Naive Bayes", "Neural Net", "Random Forest", "Recursive Partitioning and Regression Trees", "Support Vector Machines Linear", "Support Vector Machines Radial", "Bagged Classification and Regression Trees")
accuracy <- c(bayesglmAccuracy, gbmAccuracy, knnAccuracy, nbAccuracy, nnetAccuracy, rfAccuracy, rpartAccuracy, svmlAccuracy, svmrAccuracy, treebagAccuracy)
outOfSampleError <- c(bayesglmOutOfSampleError, gbmOutOfSampleError, knnOutOfSampleError, nbOutOfSampleError, nnetOutOfSampleError, rfOutOfSampleError, rpartOutOfSampleError, svmlOutOfSampleError, svmrOutOfSampleError, treebagOutOfSampleError)

results <- data.frame(trainMethods, accuracy, outOfSampleError)
results[order(results$accuracy),]
```
```
                                  trainMethods     accuracy outOfSampleError
3                           K Nearest Neighbor 0.6495327103     0.3504672897
4                                  Naive Bayes 0.7196261682     0.2803738318
8               Support Vector Machines Linear 0.7523364486     0.2476635514
7  Recursive Partitioning and Regression Trees 0.7616822430     0.2383177570
10  Bagged Classification and Regression Trees 0.7616822430     0.2383177570
2               Generalized Boosted Regression 0.7663551402     0.2336448598
6                                Random Forest 0.7803738318     0.2196261682
1                                 Bayesian GLM 0.7897196262     0.2102803738
9               Support Vector Machines Radial 0.7897196262     0.2102803738
5                                   Neural Net 0.8084112150     0.1915887850
```
---

### Model Accuracy's and Out of Sample Errors

Our selected Neural Network model has an **estimated accuracy** of **80.84%** and an **estimated out of sample error** of **0.19%**.

Looking at the confusion matrix:
```r
#Cross-validation
predictCrossVal <- predict(nnet, testing_part)
confusionMatrix(testing_part$Survived, predictCrossVal)
```
```r
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 117  10
         1  31  56
                                                
               Accuracy : 0.8084112             
                 95% CI : (0.7491985, 0.8588647)
    No Information Rate : 0.6915888             
    P-Value [Acc > NIR] : 0.00007989564         
                                                
                  Kappa : 0.5872613             
 Mcnemar's Test P-Value : 0.001787289           
                                                
            Sensitivity : 0.7905405             
            Specificity : 0.8484848             
         Pos Pred Value : 0.9212598             
         Neg Pred Value : 0.6436782             
             Prevalence : 0.6915888             
         Detection Rate : 0.5467290             
   Detection Prevalence : 0.5934579             
      Balanced Accuracy : 0.8195127             
                                                
       'Positive' Class : 0 
```

We visualise the network below:
```r
#Plot our trained Neural Network
plotnet(nnet,node_labs = TRUE,var_labs = TRUE)
```
![Neural Net](Charts/NeuralNet.jpg)

---
