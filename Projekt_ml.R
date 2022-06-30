#projekt zaliczeniowy z przedmiotu „uczenie maszynowe”
#Dzmitry Barbash i Lizaveta Kameneva

#wczytywanie bibliotek, potrzebnych dla realizacji projektu
library(rpart.plot)
library(rpart)
library(dplyr)
library(caret)
library(NeuralNetTools)
library(caret)
library(nnet)

#wczytywanie danych
PM_test <- read.csv2("PM_test.csv")
PM_train <- read.csv2("PM_train.csv")

#sprawdzamy, które dane  się powtarzają
summary(data)

#usuwamy niepotrzebne dane i normalizujemy tabelę
data <- select(PM_train,-setting3,-s1,-s2,-s5,-s6,-s7,-s8,-s10,-s11,-s12,-s13,-s15,-s16,-s18,-s19,-s20,-s21)
data_test <- select(PM_test,-setting3,-s1,-s2,-s5,-s6,-s7,-s8,-s10,-s11,-s12,-s13,-s15,-s16,-s18,-s19,-s20,-s21)
data$label3 <- as.integer(ifelse(data$RUL<40 & data$cycle<200,1,0))
data_test$label3 <- as.integer(ifelse(data_test$RUL<40 & data_test$cycle<200,1,0))
scaled_data<-as.data.frame(scale(data))
scaled_data_test<-as.data.frame(scale(data_test))
####################################################

#tworzymy model regresji liniowej
model1<-lm(RUL~.,scaled_data)
summary(model1)

#obliczmy predykcję na danych treiningowych
p1 <- predict(model1,scaled_data)

#wizualizacja
plot(scaled_data$RUL, p1)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#obliczamy korelację
cor1 <- cor(p1, scaled_data$RUL)

#sprawdzamy błąd kwadratowy
MSE.1 <- mean((scaled_data$RUL-p1)^2)



#obliczmy predykcję na danych treiningowych
p_test1 <- predict(model1,scaled_data_test)

#wizualizacja
plot(scaled_data_test$RUL, p_test1)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#obliczamy korelację
cor_test_1 <- cor(p_test1, scaled_data_test$RUL)

#sprawdzamy błąd kwadratowy
MSE.test1 <- mean((scaled_data_test$RUL-p_test1)^2)





############################################

# tworzenie modelu treiningowego(method - ranger)
model2 <- train(
  RUL ~ .,
  tuneLength = 1,
  data = scaled_data, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)
model2

#predykcja
p2<-predict(model2,scaled_data)
p2_test<-predict(model2,scaled_data_test)

#wizualizacja
plot(scaled_data$RUL, p2)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#korelacja
cor2 <- cor(p2, scaled_data$RUL)

#błąd kwadratowy
MSE.2 <- mean((scaled_data$RUL-p2)^2)

#wizualizacja danych testowych
plot(scaled_data_test$RUL, p2_test)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#korelacja
cor_test_2 <- cor(p2_test, scaled_data_test$RUL)

#błąd kwadratowy
MSE.test2 <- mean((scaled_data_test$RUL-p2_test)^2)

###
tuneGrid <- data.frame(
  .mtry = c(2, 3, 7),
  .splitrule = "variance",
  .min.node.size = 5
)

#kolejny model treiningowy
model3 <- train(
  RUL ~ .,
  tuneGrid = tuneGrid,
  data = scaled_data, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

model3

#plot
plot(model3)

#predykcja
p3<-predict(model3,scaled_data)
p3_test<-predict(model3,scaled_data_test)


#wizualizacja
plot(scaled_data$RUL, p3)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#korelacja
cor3 <- cor(p3, scaled_data$RUL)

#błąd kwadratowy
MSE.3 <- mean((scaled_data$RUL-p3)^2)

#wizualizacja na danych testowych
plot(scaled_data_test$RUL, p3_test)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

#korelacja
cor_test_3 <- cor(p3_test, scaled_data_test$RUL)

#błąd kwadratowy
MSE.test3 <- mean((scaled_data_test$RUL-p3_test)^2)

############################################################


#tworzenie modelu nnet
model4=nnet(RUL~., 
            scaled_data,
            size = 5,
            rang = 0.1,
            decay= 0.05,
            maxit= 5000)

print(model4)

#plot
plotnet(model4)
garson(model4)

#predykcja
p4<-predict(model4)
p4_test<-predict(model4,scaled_data_test)

#wizualizacja
plot(scaled_data$RUL, p4)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 3)

#korelacja
cor4 <- cor(p4, scaled_data$RUL)

#błąd kwadratowy
MSE.4 <- mean((p4-scaled_data$RUL)^2)

#wizualizacja na danych testowych
plot(scaled_data_test$RUL, p4_test)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 3)

#korelacja
cor_test_4 <- cor(p4_test, scaled_data_test$RUL)

#błąd kwadratowy
MSE.test4 <- mean((scaled_data_test$RUL-p4_test)^2)



#tworzenie modelu treiningowego z pomocą innej metody (ppr - Projection Pursuit Regression)
model5 <- train(
  RUL~., 
  scaled_data,
  method = "ppr",
  trControl = trainControl(
    method = "repeatedcv", 
    number = 3,
    repeats = 3, 
    verboseIter = TRUE
  )
)

#predykcja
p5<-predict(model5,scaled_data)
p5_test<-predict(model5,scaled_data_test)

#wizualizacja
plot(scaled_data$RUL, p5)
abline(a=0,b=1,col="red")

#korelacja
cor5<-cor(p5, scaled_data$RUL)

#błąd kwadratowy
MSE.5<-mean((scaled_data$RUL-p5)^2)

#wizualizacja na danych testowych
plot(scaled_data_test$RUL, p5_test)
abline(a=0,b=1,col="red")

#korelacja
cor_test_5<-cor(p5_test, scaled_data_test$RUL)

#błąd kwadratowy
MSE.test5<-mean((scaled_data_test$RUL-p5_test)^2)


par(mfrow = c(2, 5))

plot(scaled_data$RUL, p1,ylab="Training data",xlab=NA,main="lm")
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=3,y=-2.2,labels=paste("Cor is", format(cor1,digits=2)))
text(x=3,y=-1.8,labels=paste("MSE is", format(MSE.1,digits=2)))

plot(scaled_data$RUL, p2,xlab=NA,ylab=NA,main="TuneLength")
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=3,y=-1.2,labels=paste("Cor is", format(cor2,digits=2)))
text(x=3,y=-0.8,labels=paste("MSE is", format(MSE.2,digits=2)))

plot(scaled_data$RUL, p3,xlab=NA,ylab=NA,main="TuneGrid")
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=3,y=-1.2,labels=paste("Cor is", format(cor3,digits=2)))
text(x=3,y=-0.8,labels=paste("MSE is", format(MSE.3,digits=2)))

plot(scaled_data$RUL, p4,xlab=NA,ylab=NA,main="Nnet")
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=3,y=0.05,labels=paste("Cor is", format(cor4,digits=2)))
text(x=3,y=0.15,labels=paste("MSE is", format(MSE.4,digits=2)))

plot(scaled_data$RUL, p5,xlab=NA,ylab=NA,main="Projection Pursuit Regression")
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=3,y=-1.5,labels=paste("Cor is", format(cor5,digits=2)))
text(x=3,y=-1.1,labels=paste("MSE is", format(MSE.5,digits=2)))

plot(scaled_data_test$RUL, p_test1,ylab = "Testing data",xlab=NA)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=2.85,y=-2.6,labels=paste("Cor is", format(cor_test_1,digits=2)))
text(x=2.85,y=-2.2,labels=paste("MSE is", format(MSE.test1,digits=2)))

plot(scaled_data_test$RUL, p2_test,xlab=NA,ylab=NA)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=2.9,y=-1.2,labels=paste("Cor is", format(cor_test_2,digits=2)))
text(x=2.9,y=-0.8,labels=paste("MSE is", format(MSE.test2,digits=2)))

plot(scaled_data_test$RUL, p3_test,xlab=NA,ylab=NA)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=2.9,y=-1.2,labels=paste("Cor is", format(cor_test_3,digits=2)))
text(x=2.9,y=-0.8,labels=paste("MSE is", format(MSE.test3,digits=2)))

plot(scaled_data_test$RUL, p4_test,xlab=NA,ylab=NA)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=-1.8,y=0.8,labels=paste("Cor is", format(cor_test_4,digits=2)))
text(x=-1.8,y=0.9,labels=paste("MSE is", format(MSE.test4,digits=2)))

plot(scaled_data_test$RUL, p5_test,xlab=NA,ylab=NA)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
text(x=2.85,y=-2,labels=paste("Cor is", format(cor_test_5,digits=2)))
text(x=2.85,y=-1.6,labels=paste("MSE is", format(MSE.test5,digits=2)))

par(mfrow = c(1, 1))

#tworzymy i wizualizujemy drzewo na podstawie danych

m.rpart <-rpart(RUL ~ ., data)

rpart.plot(m.rpart, digits = 3)

#######################################
#######################################
#######################################
#WNIOSEK 
# po sprawdzeniu i porównaniu kilku modelej, ich wizualizacji,
# wyliczeniu korelacji i błędu kwadratowego zrobiliśmy wniosek, że 
# najlepszym wytrenowanym modelem jest model 5.
# na danych treiningowych najlepsze wyniki mają model 3 i model 2.


# w modelu 5 korelacja na danych treiningowych wyniosła 0.86, na testowych 0.71.
# błąd kwadratowy na danych treiningowych tylko 0.27 i na testowych 0.5
