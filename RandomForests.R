#Project:Application of machine learning algorithms to predict human environmental cadmium exposure based on urine metabolic profiles
#Title:RandomForests
#Data:November 2021

#load the libraries
library(readxl)
library(caret)
library(pROC)
library(randomForest)

#read the data
setwd("/Users/Documents/data/")
data.neg<- read_excel("neg.xlsx")

#data pre-processing
data<- data.neg[,-1]
t<- t(data)
a<- data.frame(t)
data<- a
class(data)
data$X1<- as.factor(data$X1)

#random forests
##split the data set
set.seed(100)
select_train <- sample(dim(data)[1], dim(data)[1] * 0.7)
train <- data[select_train, ]
test <- data[-select_train, ]
dim(train)
dim(test)
prop.table(table(train$class))

#RF model
forest <- randomForest(X1~.,data = train,ntree = 2000,mtry = 15)
forest
plot(forest)

#importance variables
imp<- importance(forest,type=2)
imp
varImpPlot(forest, main = "randomforests variable importance",n.var = 30,color="black",lcolor="white")
##prediction on the test set
prediction <- predict(forest, newdata = test,type = 'response',probability = TRUE)
prediction
##output confusion matrix
matrix <- confusionMatrix(test$X1,prediction)
matrix

#ROC curve
pre<- as.numeric(prediction)
roc1 <- roc(test$X1,pre)
plot(roc1, print.auc=T, auc.polygon=T, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=T,
     auc.polygon.col="skyblue", 
     print.thres=T)
##establish an empty value
auc_value<-as.numeric()

#5-fold cross-validation
set.seed(100)
folds<-createFolds(y=data$X1,k=5)
for(i in 1:5){
  test <- data[folds[[i]],] 
  train <- data[-folds[[i]],]
  fold_pre <- randomForest(X1~.,data = train,ntree = 2000,mtry = 15)
  fold_predict <- predict(fold_pre,type='response',newdata=test)
  auc_value<- append(auc_value,as.numeric(auc(as.numeric(test$X1),as.numeric(fold_predict))))
}

#AUC
auc_value
mean(auc_value)