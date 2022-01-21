library(readxl)
library(car)
library(performance)
library(see)
library(patchwork)
library(rpart)
library(rpart.plot)
library(caret)

spotifyData2=read_excel('~/Documents/DMML Project Documents/spotify_hitpredictorDatasets/SpotifySongs1990-2019.xlsx')
summary(spotifyData2)
names(spotifyData2)
str(spotifyData2)

#Data Pre-Processing
spotifyData2$hitf=as.factor(spotifyData2$target)

#Data Partitioning
set.seed(1234)
pd2=sample(2,nrow(spotifyData2), replace=TRUE, prob=c(0.7,0.3))
spotify2_train=spotifyData2[pd2==1,]
spotify2_test=spotifyData2[pd2==2,]


spotify2_tree1=rpart(hitf~danceability+energy+loudness+acousticness+instrumentalness+valence, data=spotify2_train, method="class")
summary(spotify2_tree1)
spotify2_tree1
rpart.plot(spotify2_tree1, extra=1)

#Prediction
spotify2_pred=predict(spotify2_tree1,spotify2_test, type="class")
spotify2_pred
spotify2_test$predictedrating=spotify2_pred

#Confusion Matrix

spotify2_cm=confusionMatrix(spotify2_test$hitf, spotify2_test$predictedrating)
spotify2_cm

#Cross Validation and Pruning
spotify2_bestfit=rpart(hitf~danceability+energy+loudness+acousticness+instrumentalness+valence, data=spotifyData2, method="class", parms = list(split='information')
                      , control=rpart.control(cp=0.0002,minsplit=500,minbucket = 100, maxdepth = 15,xval=10))

rpart.plot(spotify2_bestfit, cex=0.5, extra=4)
printcp(spotify2_bestfit)
plotcp(spotify2_bestfit)

#Optimum cp value: 0.00314784

spotify2_treefit=prune(spotify2_bestfit, cp=0.00314784)
rpart.plot(spotify2_treefit, cex=0.5, extra=4)

#Final Train and test results 

spotify2_finaltrain=rpart(hitf~danceability+energy+loudness+acousticness+instrumentalness+valence, data=spotify2_train, method="class", parms = list(split='information')
                         , control=rpart.control(cp=0.00314784,minsplit=500,minbucket = 100, maxdepth = 15))

spotify2_finaltrain
rpart.plot(spotify2_finaltrain, cex=0.59)

spotify2_predFinal=predict(spotify2_finaltrain, spotify2_test, type="class")
spotify2_predFinal
spotify2_test$predictedrating=spotify2_predFinal

spotify2_finalcm=confusionMatrix(spotify2_test$hitf, spotify2_test$predictedrating)
spotify2_finalcm

# library(pROC)
# spotify2_test$predictedrating=as.numeric(spotify2_test$predictedrating)
# roc(spotify2_test$hitf, spotify2_test$predictedrating)
# a2=roc(spotify2_test$hitf, spotify2_test$predictedrating)
# plot(a2,col = "#f0232b", family = "sans", cex = 2, main = "Decision Tree - ROC Curve at AUC 0.7894")


############################# RANDOM FOREST MODELS ############################################---------------------------

library(randomForest)
rf2=randomForest(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify2_train)
print(rf2)
attributes(rf2)
summary(rf2)

spotify2_rf2_trainPred=predict(rf2,spotify2_train)
head(spotify2_rf2_trainPred)
head(spotify2_train$hitf)
confusionMatrix(spotify2_rf2_trainPred,spotify2_train$hitf)

spotify2_rf2_testPred=predict(rf2,spotify2_test)
head(spotify2_rf2_testPred)
head(spotify2_test$hitf)
confusionMatrix(spotify2_rf2_testPred,spotify2_test$hitf)
plot(rf2)


##Model Tuning 1: 10 folds cross-validation
library(mlbench)
library(e1071)
control2 <- trainControl(method='repeatedcv', number=5, repeats=3)
metric <- "Accuracy"
set.seed(123)
mtry <- 1
tunegrid <- expand.grid(.mtry=mtry)
rf_default2 <- train(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence, data=spotify2_train, method='rf', metric='Accuracy', tuneGrid=tunegrid, trControl=control2)
print(rf_default2)

##Model Tuning 2: Random parameter

mtry <- 1
#ntree: Number of trees to grow.
ntree <- 3

control2_2 <- trainControl(method='repeatedcv', number=10, repeats=3,search = 'random')

#Random generate 15 mtry values with tuneLength = 15
set.seed(1)
rf_random2 <- train(hitf~danceability+energy+loudness+speechiness+acousticness+instrumentalness+valence,data=spotify2_train,method = 'rf',metric = 'Accuracy',tuneLength  = 15, trControl = control2_2)
print(rf_random2)

#Final Results

spotify2_rf2_trainPred_Final=predict(rf_default2,spotify2_train)
head(spotify2_rf2_trainPred_Final)
head(spotify2_train$hitf)
confusionMatrix(spotify2_rf2_trainPred_Final,spotify2_train$hitf)

spotify2_rf2_testPred_Final=predict(rf_default2,spotify2_test)
head(spotify2_rf2_testPred_Final)
head(spotify2_test$hitf)
confusionMatrix(spotify2_rf2_testPred_Final,spotify2_test$hitf)
plot(rf2, main = "Random Forest - Error vs Tree number")
