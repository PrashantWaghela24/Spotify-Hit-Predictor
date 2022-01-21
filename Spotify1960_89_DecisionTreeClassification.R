library(readxl)
library(car)
library(performance)
library(see)
library(patchwork)
library(rpart)
library(rpart.plot)
library(caret)

spotifyData=read_excel('~/Documents/DMML Project Documents/spotify_hitpredictorDatasets/SpotifySongs1960-89.xlsx')
summary(spotifyData)
names(spotifyData)
str(spotifyData)

#Data Pre-Processing
spotifyData$hitf=as.factor(spotifyData$target)
spotifyData=spotifyData[-c(1:3,15)]
str(spotifyData)

#Data Partitioning
set.seed(1234)
pd=sample(2,nrow(spotifyData), replace=TRUE, prob=c(0.7,0.3))
spotify_train=spotifyData[pd==1,]
spotify_test=spotifyData[pd==2,]

spotify_tree1=rpart(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify_train, method="class")
rpart.plot(spotify_tree1, extra=1)
summary(spotify_tree1)
#Prediction
spotify_pred=predict(spotify_tree1,spotify_test, type="class")
spotify_pred
spotify_test$predictedrating=spotify_pred

#Confusion Matrix
spotify_cm=confusionMatrix(spotify_test$hitf, spotify_test$predictedrating)
spotify_cm

#Cross Validation and Pruning
spotify_bestfit=rpart(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotifyData, method="class", parms = list(split='information')
                      , control=rpart.control(cp=0.0002,minsplit=500,minbucket = 100, maxdepth = 15, xval=10))

rpart.plot(spotify_bestfit, cex=0.5, extra=4)
printcp(spotify_bestfit)
plotcp(spotify_bestfit)

#Optimum cp value:  0.00180134

spotify_treefit=prune(spotify_bestfit, cp=0.00180134)
rpart.plot(spotify_treefit, cex=0.5, extra=4)

#Final Train and test results 

spotify_finaltrain=rpart(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify_train, method="class", parms = list(split='information')
                         , control=rpart.control(cp=0.00180134,minsplit=500,minbucket = 100, maxdepth = 15))

spotify_finaltrain
rpart.plot(spotify_finaltrain, cex=0.59)

spotify_predFinal=predict(spotify_finaltrain, spotify_test, type="class")
spotify_predFinal
spotify_test$predictedrating=spotify_predFinal
library(caret)
spotify_finalcm=confusionMatrix(spotify_test$hitf, spotify_test$predictedrating)
spotify_finalcm
# library(pROC)
# spotify_test$predictedrating=as.numeric(spotify_test$predictedrating)
# roc(spotify_test$hitf, spotify_test$predictedrating)
# a=roc(spotify_test$hitf, spotify_test$predictedrating)
# plot(a,col = "#f0232b", family = "sans", cex = 2, main = "Decision Tree - ROC Curve at AUC 0.7193")

############################# RANDOM FOREST MODELS ############################################---------------------------

library(randomForest)
rf=randomForest(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify_train)
print(rf)
attributes(rf)
summary(rf)

spotify_rf_trainPred=predict(rf,spotify_train)
head(spotify_rf_trainPred)
head(spotify_train$hitf)
confusionMatrix(spotify_rf_trainPred,spotify_train$hitf)

spotify_rf_testPred=predict(rf,spotify_test)
head(spotify_rf_testPred)
head(spotify_test$hitf)
confusionMatrix(spotify_rf_testPred,spotify_test$hitf)

#Error Rate for Random Forest
plot(rf)

##Model Tuning 1: 10 folds cross-validation
library(mlbench)
library(e1071)
control <- trainControl(method='repeatedcv', number=5, repeats=3)
metric <- "Accuracy"
set.seed(123)
mtry <- 1
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify_train, method='rf', metric='Accuracy', tuneGrid=tunegrid, trControl=control)
print(rf_default)

##Model Tuning 2: Random parameter search

mtry <- 1
ntree <- 3

control_2 <- trainControl(method='repeatedcv', number=10, repeats=3,search = 'random')

#Random generate 15 mtry values with tuneLength = 15
set.seed(1)
rf_random <- train(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence,data=spotify_train,method = 'rf',metric = 'Accuracy',tuneLength  = 15, trControl = control_2)
print(rf_random)

#Final Results

spotify_rf_trainPred_Final=predict(rf_default,spotify_train)
head(spotify_rf_trainPred_Final)
head(spotify_train$hitf)
confusionMatrix(spotify_rf_trainPred_Final,spotify_train$hitf)

spotify_rf_testPred_Final=predict(rf_default,spotify_test)
head(spotify_rf_testPred_Final)
head(spotify_test$hitf)
confusionMatrix(spotify_rf_testPred_Final,spotify_test$hitf)
plot(rf, main = "Random Forest - Error vs Tree number")


