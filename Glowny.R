library(caret)
library(e1071)
library(rstudioapi)
library(fastDummies)
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("funkcje.R")

set.seed(666)
seed <- 666

parTune_knn = expand.grid(k = seq(3, 15, by = 3))
parTune_tree = expand.grid(type=c("Gini", "Entropy"), depth=3:6, minobs=3)
parTune_tree_rm <- expand.grid(type=c("gini", "information"), depth=3:6, minobs=3)
parTune_tree_reg <- expand.grid(type=c("SS"), depth=3:7, minobs=3)
parTune_nn<- expand.grid(iter=c(20, 50, 100), lr=c(0.1, 0.09, 0.08, 0.07))


################################################################################
################klasyfikacja binarna - przygotowanie danych#####################
################################################################################
data_bin <- read.csv("wholesale_customers.csv", sep=",")
head(data_bin)

data_bin[,1][data_bin[,1] == 2] <- 0
data_bin[,1] <- as.factor(data_bin[,1])
head(data_bin)

data_bin_norm <- data_bin
data_bin_norm[, 2:8]  <- scale(data_bin_norm[, 2:8])

#####################binarna zaimplementowane algorytmy#########################
cv_res_bin_knn <- CrossValidation(data_bin, 5, parTune_knn, seed, model='KNN')
bestKNN  <- SelectBestModelCV(cv_res_bin_knn, "KNN")

cv_res_bin_tree <- CrossValidation(data_bin, 5, parTune_tree, seed, model='TREE')
bestTree <- SelectBestModelCV(cv_res_bin_tree, "TREE")

cv_res_bin_nn <- CrossValidation(data_bin_norm, 5, parTune_nn, seed, model='NN')
bestNN <- SelectBestModelCV(cv_res_bin_nn, "NN")

models_bin_im <- rbind(rbind(bestKNN, bestTree), bestNN)

##########################binarna gotowe algorytmy##############################
cv_res_bin_knn_rm <- CrossValidationReadyModels(data_bin, 5, parTune_knn, seed, model='KNN')
bestKNN_rm  <- SelectBestModelCV(cv_res_bin_knn_rm, "KNN_rm")

cv_res_bin_tree_rm <- CrossValidationReadyModels(data_bin, 5, parTune_tree_rm, seed, model='TREE')
bestTree_rm  <- SelectBestModelCV(cv_res_bin_tree_rm, "TREE_rm")

cv_res_bin_nn_rm <- CrossValidationReadyModels(data_bin_norm, 5, parTune_nn, seed, model='NN')
bestNN_rm  <- SelectBestModelCV(cv_res_bin_nn_rm, "NN_rm")

models_bin_rm <- rbind(rbind(bestKNN_rm, bestTree_rm), bestNN_rm)

models_bin <- rbind(models_bin_im, models_bin_rm)
SelectBestModel(models_bin)

#####################wykresy implementowane algorytmy###########################
ggplot(cv_res_bin_knn, aes(k)) + 
  geom_point(aes(y = AUCW, colour = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'k' na AUC w zaimplementowanym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_bin_tree, aes(depth)) + 
  geom_point(aes(y = AUCW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = AUCT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'depth' oraz 'type' na AUC w zaimplementowanym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_bin_nn, aes(lr)) + 
  geom_point(aes(y = AUCW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = iter, shape = "Treing")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na AUC w zaimplementowanym modelu NN") +
  labs(colour="Iteracje", shape="Dane")

#########################wykresy gotowe algorytmy###############################
ggplot(cv_res_bin_knn_rm, aes(k)) + 
  geom_point(aes(y = AUCW, colour = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'k' na AUC w gotowym modelu KNN") +
  labs(colour="Legenda")

ggplot(cv_res_bin_tree_rm, aes(depth)) + 
  geom_point(aes(y = AUCW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = AUCT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'depth' oraz 'type' na AUC w gotowym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_bin_nn_rm, aes(lr)) + 
  geom_point(aes(y = AUCW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = AUCT, colour = iter, shape = "Trening")) +
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na AUC w gotowym modelu NN") +
  labs(colour="Iteracje", shape="Dane")


################################################################################
#############klasyfikacja wieloklasowa - przygotowanie danych###################
################################################################################
data_mult <- read.csv("accent.csv", sep=",", dec = ".")
data_mult[,1] <- as.factor(data_mult[,1] )

data_mult_norm <- data_mult
data_mult_norm[, 2:13]  <- scale(data_mult_norm[, 2:13])

##################wieloklasowa zaimplementowane algorytmy#######################
cv_res_mult_knn <- CrossValidation(data_mult, 5, parTune_knn, seed, model='KNN')
bestKNN_mult <- SelectBestModelCV(cv_res_mult_knn, "KNN")

cv_res_mult_tree <- CrossValidation(data_mult, 5, parTune_tree, seed, model='TREE')
bestTree_mult <- SelectBestModelCV(cv_res_mult_tree, "TREE")

cv_res_mult_nn <- CrossValidation(data_mult_norm, 5, parTune_nn, seed, model='NN')
bestNN_mult <- SelectBestModelCV(cv_res_mult_nn, "NN")

models_mult_im <- rbind(rbind(bestKNN_mult, bestTree_mult), bestNN_mult)

#######################wieloklasowa gotowe algorytmy############################
cv_res_mult_knn_rm <- CrossValidationReadyModels(data_mult, 5, parTune_knn, seed, model='KNN')
bestKNN_mult_rm  <- SelectBestModelCV(cv_res_mult_knn_rm, "KNN_rm")

cv_res_mult_tree_rm <- CrossValidationReadyModels(data_mult, 5, parTune_tree_rm, seed, model='TREE')
bestTree_mult_rm  <- SelectBestModelCV(cv_res_mult_tree_rm, "TREE_rm")

cv_res_mult_nn_rm <- CrossValidationReadyModels(data_mult_norm, 5, parTune_nn, seed, model='NN')
bestNN_mult_rm  <- SelectBestModelCV(cv_res_mult_nn_rm, "NN_rm")

models_mult_rm <- rbind(rbind(bestKNN_mult_rm, bestTree_mult_rm), bestNN_mult_rm)

models_mult <- rbind(models_mult_im, models_mult_rm)
SelectBestModel(models_mult)

#####################wykresy implementowane algorytmy###########################
ggplot(cv_res_mult_knn, aes(k)) + 
  geom_point(aes(y = JakoscW, colour = "Walidacja")) + 
  geom_point(aes(y = JakoscT, colour = "Trening")) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'k' na jakoœæ w zaimplementowanym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_mult_tree, aes(depth)) + 
  geom_point(aes(y = JakoscW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = JakoscT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'depth' oraz 'type' na jakoœæ w zaimplementowanym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_mult_nn, aes(lr)) + 
  geom_point(aes(y = JakoscW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = JakoscT, colour = iter, shape = "Trening")) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na jakoœæ w zaimplementowanym modelu NN") +
  labs(colour="Iteracje", shape="Dane")

#########################wykresy gotowe algorytmy###############################
ggplot(cv_res_mult_knn_rm, aes(k)) + 
  geom_point(aes(y = JakoscW, colour = "Walidacja")) + 
  geom_point(aes(y = JakoscT, colour = "Trening")) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'k' na jakoœæ w gotowym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_mult_tree_rm, aes(depth)) + 
  geom_point(aes(y = JakoscW, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = JakoscT, shape = "Trening", colour = type)) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'depth' oraz 'type' na jakoœæ w gotowym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_mult_nn_rm, aes(lr)) + 
  geom_point(aes(y = JakoscW, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = JakoscT, colour = iter, shape = "Trening")) +
  ylim(0,1) +
  ylab("Jakoœæ") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na jakoœæ w gotowym modelu NN") +
  labs(colour="Iteracje", shape="Dane")


################################################################################
######################regresja - przygotowanie danych###########################
################################################################################
data_reg <- read.csv("winequality-white.csv", sep=";", dec = ".")
head(data_reg)
data_reg <- data_reg[sample(nrow(data_reg), 500), ]
head(data_reg)
data_reg <- cbind("quality"=as.numeric(data_reg[,12]), data_reg[,1:11])
head(data_reg)

data_reg_norm <- data_reg
data_reg_norm[, 2:12]  <- scale(data_reg_norm[, 2:12])

######################regresja zaimplementowane algorytmy#######################
cv_res_reg_knn <- CrossValidation(data_reg, 5, parTune_knn, seed, model='KNN')
bestKNN_reg <- SelectBestModelCV(cv_res_reg_knn, "KNN")

cv_res_reg_tree <- CrossValidation(data_reg, 5, parTune_tree_reg, seed, model='TREE')
bestTree_reg <- SelectBestModelCV(cv_res_reg_tree, "TREE")

cv_res_reg_nn <- CrossValidation(data_reg_norm, 5, parTune_nn, seed, model='NN')
bestNN_reg <- SelectBestModelCV(cv_res_reg_nn, "NN")

models_reg_im <- rbind(rbind(bestKNN_reg, bestTree_reg), bestNN_reg)

###########################regresja gotowe algorytmy############################
cv_res_reg_knn_rm <- CrossValidationReadyModels(data_reg, 5, parTune_knn, seed, model='KNN')
bestKNN_reg_rm  <- SelectBestModelCV(cv_res_reg_knn_rm, "KNN_rm")

cv_res_reg_tree_rm <- CrossValidationReadyModels(data_reg, 5, parTune_tree_reg, seed, model='TREE')
bestTree_reg_rm  <- SelectBestModelCV(cv_res_reg_tree_rm, "TREE_rm")

cv_res_reg_nn_rm <- CrossValidationReadyModels(data_reg_norm, 5, parTune_nn, seed, model='NN')
bestNN_reg_rm  <- SelectBestModelCV(cv_res_reg_nn_rm, "NN_rm")

models_reg_rm <- rbind(rbind(bestKNN_reg_rm, bestTree_reg_rm), bestNN_reg_rm)

models_reg <- rbind(models_reg_im, models_reg_rm)
SelectBestModel(models_reg)

#####################wykresy implementowane algorytmy###########################
ggplot(cv_res_reg_knn, aes(k)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'k' na MAPE w zaimplementowanym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_reg_tree, aes(depth)) + 
  geom_point(aes(y = MAPEw, shape = "Walidacja", colour = type)) + 
  geom_point(aes(y = MAPEt, shape = "Trening", colour = type)) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'depth' oraz 'type' na MAPE w zaimplementowanym modelu TREE") +
  labs(shape="Dane", colour="Typ")

ggplot(cv_res_reg_nn, aes(lr)) + 
  geom_point(aes(y = MAPEw, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = iter, shape = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na MAPE w zaimplementowanym modelu NN") +
  labs(colour="Iteracje", shape="Dane")

#########################wykresy gotowe algorytmy###############################
ggplot(cv_res_reg_knn_rm, aes(k)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'k' na MAPE w gotowym modelu KNN") +
  labs(colour="Dane")

ggplot(cv_res_reg_tree_rm, aes(depth)) + 
  geom_point(aes(y = MAPEw, colour = "Walidacja")) + 
  geom_point(aes(y = MAPEt, colour = "Train" )) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'depth' na MAPE w gotowym modelu TREE") +
  labs(colour="Dane")

ggplot(cv_res_reg_nn_rm, aes(lr)) + 
  geom_point(aes(y = MAPEw, colour = iter, shape = "Walidacja")) + 
  geom_point(aes(y = MAPEt,colour = iter, shape = "Trening")) +
  ylim(0,2) +
  ylab("MAPE") +
  ggtitle("Wp³yw 'lr' oraz 'iter' na MAPE w gotowym modelu NN") +
  labs(colour="Iteracje", shape="Dane")
