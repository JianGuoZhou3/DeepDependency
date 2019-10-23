#!/usr/bin/R

options(stringsAsFactors=F)
args = commandArgs(trailingOnly =T)
require(ROCR)
library(gplots)

#Test
temp = list.files(path="./out", pattern="Test*", full.names=TRUE)
pred.ls   = list()
labels.ls = list()
for (i in 1:5) {
        pred.ls[[i]]   = as.matrix(read.table(temp[i]))[,2]
        labels.ls[[i]] = as.matrix(read.table(temp[i]))[,1]
}
RF_ROC_predict  = prediction(predictions=pred.ls, labels=labels.ls)
test            = performance(RF_ROC_predict, measure="tpr", x.measure="fpr")
AUC             = performance(RF_ROC_predict, measure="auc")
test_AUC        = round(mean(as.numeric(AUC@y.values)),3)

#Train
temp = list.files(path="./out", pattern="Train*", full.names=TRUE)
pred.ls   = list()
labels.ls = list()
for (i in 1:5) {
        pred.ls[[i]]   = as.matrix(read.table(temp[i]))[,2]
        labels.ls[[i]] = as.matrix(read.table(temp[i]))[,1]
}
RF_ROC_predict  = prediction(predictions=pred.ls, labels=labels.ls)
train           = performance(RF_ROC_predict, measure="tpr", x.measure="fpr")
AUC             = performance(RF_ROC_predict, measure="auc")
train_AUC       = round(mean(as.numeric(AUC@y.values)),3)


pdf("./out/Model_AUC.pdf")
par(mar=c(5,5,5,5))
plot(test, avg="threshold", spread.estimate="stderror", col="navyblue", lwd=3, xlab="", ylab="")
plot(train, avg="threshold", spread.estimate="stderror", col="orange", add=T, lwd=3)
abline(a=0,b=1,lty=2, lwd=3)
title(main="Model AUC", xlab="True positive rate",ylab="False positive rate", cex.main =2, cex.lab=1.5)
legend("bottomright", c(paste("Test AUC=    ", test_AUC), paste("Training AUC=", train_AUC)), col=c("navyblue","orange"), lty=c(1,1), pch=22,lwd=3, cex=0.9)





