######## 1 -way Anova ###########
data<-read.csv(file.choose())   #D:\Downloads\casestudy1.csv
head(data)
anova_1way<-aov(data$.MWS~factor(data$MN))
summary(anova_1way)
Tukey_test_1way<-TukeyHSD(anova_1way)
Tukey_test_1way

######## 2-way annova ################
data<-read.csv(file.choose())
head(data)
annova_2way<-aov(data$.MWS~factor(data$MN)+factor(data$HR))
summary(annova_2way)
Tukey_test_2way<-TukeyHSD(annova_2way)
Tukey_test_2way

######## 1 -way MANOVA ###############
data<-iris
y<-cbind(data$Sepal.Length,data$Sepal.Width,data$Petal.Length,data$Petal.Width)
y

process<-gl(3,50,labels = c("Setosa","Verginica","Vercicolor"))

manova_1way<-manova(y~process)
summary(manova_1way)

##### Gradient-Descent Linear Regression ################
x<-c(80,100,120,140,160,180,200,220,240,260)
y<-c(70,65,90,95,110,115,120,140,155,150)
x_cal<-(x-min(x))/(max(x)-min(x))
y_cal<-(y-min(y))/(max(y)-min(y))

x_cal<-cbind(1,matrix(x_cal))
theta<-matrix(c(0,0),nrow = 2)
alpha=0.01
iteration<-10000

for(i in 1:iteration){
  error<- (x_cal %*% theta - y_cal)
  delta<- t(x_cal) %*% error /length(y_cal)
  theta<- theta- delta * alpha
}
theta
glm.fit<-lm(y_cal~x_cal)
print(glm.fit)



####### Gradient Descent Logistic Regression ################
data<-read.csv(file.choose())
head(data)
glm.fit<-glm(data$label~data$score.1+data$score.2,family = "binomial")
summary(glm.fit)
coef(glm.fit)

x<-as.matrix(data[,c(1,2)])
x<-cbind(rep(1,nrow(x)),x)
y<-as.matrix(data$label)
inital_theta<-rep(0,ncol(x))

sigmoid<-function(z){
  g<-1/(1+exp(-z))
  return(g)
}

cost<-function(theta){
  m<-nrow(x)
  g<-sigmoid(x %*% theta)
  J = (1/m) * sum(-y*log(g)-((1-y)*log(1-g)))
  return(J)
}

cost(inital_theta)

theta_optim<-optim(par = inital_theta,fn=cost)
theta_optim$value


######### Subset ###############

ibrary(MASS)
data("Boston")
dim(Boston)
fix(Boston)
head(Boston)
sum(is.na(Boston$medv))
install.packages("leaps")
library(leaps)
lm.fit<-lm(medv~.,Boston)
summary(lm.fit)
regfit.full=regsubsets(medv~.,data=Boston,nvmax=13)
reg.summary=summary(regfit.full)
reg.summary
names(reg.summary)
reg.summary$rsq
which.max(reg.summary$adjr2)
which.min(reg.summary$cp)
which.min(reg.summary$bic)
coef(regfit.full,11)


##### LDA IRIRS ######
data<-iris
library("MASS")
library("caTools")
sample = sample.split(data,SplitRatio = 0.75)
train = subset(data,sample==TRUE)
test = subset(data,sample==FALSE)

lda.fit<-lda(factor(train$Species)~.,data=train)
predict_train<-predict(lda.fit,train)
table(predict_train$class,train$Species)

lda.fit2<-lda(factor(test$Species)~.,data=test)
predict_test<-predict(lda.fit2,test)
table(predict_test$class,test$Species)


########### PCA IRIRS ############
data<-iris
library(nnet)
fit<-multinom(data$Species~.,data = data)
summary(fit)
predict<-predict(fit)
table(predict,data$Species)

fit2<-lda(factor(data$Species)~.,data=data)
summary(fit2)
predict_lda<-predict(fit2,data)
table(predict_lda$class,data$Species)



m<-matrix(c(data$Sepal.Length,data$Sepal.Width,data$Petal.Length,data$Petal.Width),ncol = 4)
p<-prcomp(m)
p$sdev #eignen values
p$rotation #eigen vectors
p$x
label<-p$x
label<-data.frame(label)
label1<-cbind(label$PC1,label$PC2,label$PC3,label$PC4,data$Species)
label1<-data.frame(label1)

fit3<-multinom(label1$X5~.,data=label1)
predict_pca_mulnom<-predict(fit3)
table(predict_pca_mulnom,label1$X5)

fit4<-lda(label1X5.Species~.,data=label1)
summary(fit4)
predict_pca_lda<-predict(fit4,label1)
table(predict_pca_lda$class,label1$X5)


############### KNN ##############################
data<-iris
control<-trainControl(method = "cv",number = 10)
model_knn<-train(Species~.,data = data,method="knn",trControl=control,tuneLength=5)
model_knn$results

########### K- Means #####################
data<-iris
data<-data[-c(5)]
fit<-kmeans(data,3)

fit$centers
fit$size
fit$cluster
 ######### Hierarchical ###############
data<-iris
data<-data[-c(5)]
z<-data
h<-hclust(dist(z),method = "average") # , labels=c()
h$height
plot(h)



########## SVM ####################


data<-iris
data$Species<-as.factor(data$Species)
data
svmfit<-svm(Species~.,data = data,kernel="linear",cost=100000,tot=0.00000001,shrinking=FALSE,scale = FALSE)
pred<-predict(svmfit)
table(pred,data$Species)
