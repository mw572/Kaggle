### KAGGLE - TITANIC SURVIVAL - 02/09/15 - Marcus Williamson ###

## clear up environment
ls()
rm(list=ls())


## import libraries

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

registerDoMC(cores = 8) # set cores for parallel processing

set.seed(8484) # set our seed for reproducibility


## importing our data from file
rawdata = read.csv("train.csv", header = TRUE)


## initial look at our data
dim(rawdata)
View(rawdata)


## removing obsolete variables from datasets
surv <- rawdata$Survived # keeping our survival variables aside
id <- rawdata$PassengerId # keeping passenger id

names <- grepl("Name|Ticket|Cabin|PassengerId", colnames(rawdata)) # getting all non predictive variables column names

rawdata = rawdata[,!names] # removing columns


## futher cleansing of data incompatible with ML problems (na values)
data = rawdata[complete.cases(rawdata),]

data$Survived <- factor(data$Survived)

## partitioning training data into training and validation datasets
inTrain <- createDataPartition(y=data$Survived,p=0.70, list=FALSE) # using a 70:30 split

training_part <- data[inTrain,] # 70% of data for training
testing_part <- data[-inTrain,] # 30% of data for later validation

# create an empty numeric vector to calculate out of sample error against
outOfSampleError <- numeric()

# add some parameters for train control
TC <- trainControl(method = "cv", number = 12, returnData=FALSE, returnResamp="none", savePredictions=FALSE, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)

plot(training_part$Survived, col="blue", main="Bar Plot of levels of the variable Survived within the sub data set", xlab="Survived levels", ylab="Frequency")

#Build Models
# train, predict, calculate accuracy and out of sample error
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


#Cross-validation
predictCrossVal <- predict(nnet, testing_part)
confusionMatrix(testing_part$Survived, predictCrossVal)

#Plot our trained Neural Network
plotnet(nnet,node_labs = TRUE,var_labs = TRUE)