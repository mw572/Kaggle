### Kaggle Script - Titanic Survival Model

## clear up environment
ls()
rm(list=ls())

setwd("~/Documents/R/Kaggle/Titanic Survival Model")

## import libraries
library(caret)
library(rpart)
library(doMC)
library(rpart.plot)

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

training <- data[inTrain,] # 70% of data for training
testing <- data[-inTrain,] # 30% of data for later validation

dim(training);dim(testing) # ensuring we have sufficient volumes for this 70:30 split

## creating our model
#We are using a Random Forest model with K fold Cross Validation with 10 folds , we use 300 trees in the model training

modFit <- train(Survived ~ ., data=training,method="rf", trControl=trainControl(method="cv", number=5), verbose=FALSE, ntree=200, allowParallel=TRUE)
modFit # examining the model
modFit$finalModel

## cross looking at the predictive ability of our model
prediction <- predict(modFit, testing)
confusionMatrix(testing$Survived, prediction) # printing error matrix

accuracy <- postResample(prediction, testing$Survived) # calculating accuracy
ose <- 1 - as.numeric(confusionMatrix(testing$Survived, prediction)$overall[1]) # calculating out of sample error

accuracy;ose # printing these values


## using the model to predict on the unseen data
results <- predict(modFit, testdata[, -length(names(testdata))])
results


## plotting random forest model output
plot(modFit$finalModel,main="Log of resampling results across tuning parameters", log="y")
finalmodel.legend <- if (is.null(modFit$finalModel$test$err.rate)) {colnames(modFit$finalModel$err.rate)} else {colnames(modFit$finalModel$test$err.rate)}
legend("top", cex =0.5, legend=finalmodel.legend, lty=c(1,2,3,4,5,6), col=c(1,2,3,4,5,6), horiz=T) # plotting rival model error rates from which the final model was selected

varImpPlot(modFit$finalModel, main="Importance Plot") # plotting the relative importance of the variables in the final model

treeout <- rpart(Survived ~ ., data=training) # creating a single tree from the training data
prp(treeout,tweak=1) # plotting a single sample tree as Random Forest is black box algorithm, a single tree does not represent the model but is useful for sense checking data
