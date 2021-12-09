#Project:Application of machine learning algorithms to predict human environmental cadmium exposure based on urine metabolic profiles
#Title:XGBoost
#Data:November 2021

#read the data
setwd("/Users/Documents/data/")
library(readxl)
data.neg<- read_excel("neg.xlsx")

#data pre-processing
data<- data.neg[,-1]
t<- t(data)
a<- data.frame(t)
data<- a
class(data)

#xgboost
#load the libraries
library(caret)
library(xgboost)
library(Matrix)
library(pROC)
##split the data set
select_train <- sample(dim(data)[1], dim(data)[1] * 0.7)
train <- data[select_train, ]
test <- data[-select_train, ]

#The xgbDMatrix object required to construct the model is processed with a sparse matrix
traindata1 <- data.matrix(train[,c(2:3559)]) 
traindata2 <- Matrix(traindata1,sparse=T) 
traindata3 <- train[,1]
traindata4 <- list(data=traindata2,label=traindata3) 
dtrain <- xgb.DMatrix(data = traindata4$data, label = traindata4$label) 

#xgboost model
xgb <- xgboost(data = dtrain,max_depth=6, eta=0.5,  objective='binary:logistic',nround=25)
####The data for the test set is processed as a matrix
testset1 <- data.matrix(test[,c(2:3559)]) 
testset2 <- Matrix(testset1,sparse=T) 
testset3 <- test[,1]
testset4 <- list(data=testset2,label=testset3) 
dtest <- xgb.DMatrix(data = testset4$data, label = testset4$label) 

#importance 
model <- xgb.dump(xgb, with_stats = T)
model[1:30] 
#This statement prints top 30 nodes of the model
names <- dimnames(data.matrix(train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30,],main="xgboost variable importance",col="black",cex=0.9)
#Predict on the test set
pre_xgb <- round(predict(xgb,newdata = dtest))
#Output confusion matrix
table(test$X1,pre_xgb,dnn=c("True","Predict"))
pre <- round(predict(xgb,newdata = dtest,type='response'))
class(pre)
matrix <- confusionMatrix(as.factor(pre),as.factor(test$X1))
matrix

#ROC curve
xgboost_roc <- roc(test$X1,as.numeric(pre_xgb))
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='ROC curve')

#establish an empty value
auc_xg<-as.numeric()

#5-fold cross-validation
folds <- createFolds(y=data$X1,k=5)
for(i in 1:5){
  test <- data[folds[[i]],] 
  train <- data[-folds[[i]],]
  traindata1 <- data.matrix(train[,c(2:3559)]) 
  traindata2 <- Matrix(traindata1,sparse=T) 
  traindata3 <- train[,1]
  traindata4 <- list(data=traindata2,label=traindata3)
  dtrain <- xgb.DMatrix(data = traindata4$data, label = traindata4$label)
  testset1 <- data.matrix(test[,c(2:3559)]) 
  testset2 <- Matrix(testset1,sparse=T) 
  testset3 <- test[,1]
  testset4 <- list(data=testset2,label=testset3) 
  dtest <- xgb.DMatrix(data = testset4$data, label = testset4$label) 
  xg_fold <- xgboost(data = dtrain,
                     max_depth=6,
                     eta=0.5,
                     objective='binary:logistic',
                     nround=25)
  pre_xg <- round(predict(xg_fold,newdata = dtest))
  auc_xg<- append(auc_xg,as.numeric(auc(as.numeric(test$X1),as.numeric(pre_xg))))
}

#AUC
auc_xg
mean(auc_xg)