# Files pml-training.csv and ml-testing.csv have been already downladed to 
# the working directory
# "NA","#DIV/0!","" values are considered missing values at data loading
pml_data <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
pml_newdata <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
# Output variable is "problem_id" in pml-testing, name is set to "classe"
colnames(pml_newdata)[colnames(pml_newdata)=="problem_id"] <- "classe"
# NA cleaning, predictors are removed
NAcols <- apply(pml_data, 2, function(x) sum(is.na(x)))
pml_data <- pml_data[,names(NAcols[NAcols==0])]
pml_newdata <- pml_newdata[,names(NAcols[NAcols==0])]
# Removing first 7 predictors; not relevant to this analysis 
pml_data <- pml_data[,8:60]
pml_newdata <- pml_newdata[,8:60]
# pml_data is splitted into training and testing datasets
library(caret)
set.seed(2425)
inTrain <- createDataPartition(y=pml_data$classe, p=0.7, list=FALSE)
pml_training <- pml_data[inTrain,]
pml_testing <- pml_data[-inTrain,]
# Predictors from NZV and high correlation analysis are removed
nearzerovariance <- nearZeroVar(pml_training[,-53], saveMetrics = TRUE)
correlationMatrix <- cor(pml_training[,-53])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
names(pml_testing[,highlyCorrelated])
pml_training <- pml_training[,-highlyCorrelated]
pml_testing <- pml_testing[,-highlyCorrelated]
pml_newdata <- pml_newdata[,-highlyCorrelated]


# Models' training and testing. Models are saved to RDS files for later use. 
# Model M01
set.seed(12125)
modelfit <- train(classe ~ ., dat = pml_training, method = "knn", preProcess=c("pca"), 
                  trControl = trainControl(method = 'cv', number = 10))
saveRDS(modelfit, "modelfitKNN-1.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M02
set.seed(12125)
modelfit <- train(classe ~ ., dat = pml_training, method = "knn", preProc=c("center", "scale"), 
                  trControl = trainControl(method = 'cv', number = 10))
saveRDS(modelfit, "modelfitKNN-2.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M03
set.seed(12125)
modelfit <- train(classe ~ ., dat = pml_training, method = "knn", preProc=c("pca", "center", "scale"), 
                  trControl = trainControl(method = 'cv', number = 10))
saveRDS(modelfit, "modelfitKNN-3.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M04
set.seed(12125)
modelfit <- train(classe ~ ., data=pml_training, method="rpart", 
                  trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitRPART-1.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M05
set.seed(12125)
modelfit <- train(classe ~ ., data=pml_training, method="rpart", 
                  trControl = trainControl(method = 'cv', number = 10), preProc = c("pca"))
saveRDS(modelfit, "modelfitRPART-2.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M06
set.seed(12125)
modelfit <- train(classe ~ ., data=pml_training, method="rf",
                  trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitRF-1.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M07
set.seed(12125)
modelfit <- train(classe ~ ., data=pml_training, method="rf")
saveRDS(modelfit, "modelfitRF-2.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M08
set.seed(12125)
modelfitSVM_1 <- train(classe ~ ., data = pml_training, method = "svmRadial", tuneLength = 1,
                       trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitSVM-1.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M09
set.seed(12125)
modelfitSVM_2 <- train(classe ~ ., data = pml_training, method="svmRadial", tuneLength = 5, 
                       trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitSVM-2.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M10
set.seed(12125)
modelfitSVM_3 <- train(classe ~ ., data = pml_training, method="svmRadial", tuneLength = 9, 
                       trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitSVM-3.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M11
set.seed(12125)
modelfitSVM_4 <- train(classe ~ ., data = pml_training, method="svmRadial", tuneLength = 15, 
                       trControl = trainControl(method = 'cv', number = 10), preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitSVM-4.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Model M12
set.seed(12125)
modelfitSVM_5 <- train(classe ~ ., data = pml_training, method="svmRadial", tuneLength = 3, 
                       preProc = c("center", "scale"))
saveRDS(modelfit, "modelfitSVM-5.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
modelfit$times[1]
cf_matrix$overall["Accuracy"]

# Applying selected models with pml-testing (pml_newdata) data 

# Model M02
modelfit <- readRDS("modelfitKNN-2.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
predict(modelfit,newdata=pml_newdata)
# [1] B A C A A B D B A A D C B A E E A B B B
# Levels: A B C D E

# Model M06
modelfit <- readRDS("modelfitRF-1.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
# This is the selected model for submitting answers
answers <- predict(modelfit,newdata=pml_newdata)
answers <- as.character(answers)
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E

# Model M11
modelfit <- readRDS("modelfitSVM-4.RDS")
prediction <- predict(modelfit,pml_testing)
cf_matrix <- confusionMatrix(prediction, pml_testing$classe)
predict(modelfit,newdata=pml_newdata)
# [1] B A B A A B D B A A B C B A E E A B B B
# Levels: A B C D E

# Write files from answers
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)