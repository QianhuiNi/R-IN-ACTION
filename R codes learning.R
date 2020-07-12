rm(list = ls())
getwd()
setwd("/Users/qianhuini/Documents")
options(digits = 3)                                  #修改设置，显示小数点后三位

#Library package
library()                                            #显示库中有哪些包
search()                                             #哪些包已加载并可使用
help(package="package_name")                         #输出某个包的简短描述以及包中的函数名称和数据集名称的列表
library(psych)
library(ggplot2)
library(rstatix)
library(dplyr)
library(plyr)
library(WRS)

#读入excel数据
library(gdata)
read.xls
HW <- read.xls(file.choose())
#读入csv数据
HW6_pt1_results <- read.csv("HW6-pt1-results2020.csv")
#写入csv
write.csv(q4_mean,"/Users/qianhuini/Desktop/File Name.csv")
#读入spss data
library(foreign) 
wais = read.spss(file.choose(), to.data.frame=TRUE)
#从带分隔符的文本文件中导入数据
read.table("xxx.csv", header = TRUE, sep = ",", row.name="name")

#建立一个空向量
x <- vector()
#建立一个向量A，其中X重复n次
A <- rep(X,n)
#type of variables
str(v)

a[c(2, 4)]                                            #访问向量a中的第2个和第4个元素
a[3]                                                  #访问向量a中的第3个元素
a[2:4]                                                #访问向量a中的第2个至第4个元素

#判断元素是否在向量内
2 %in% c(2,3,5,6)
c(1, 2) %in% c(2, 3, 5, 6)

#turning x into a factor variable
female$race_cat2 <- factor(female$race , labels = c("Asian" , "Black" , "Hispanic" , 
                                                    "White" , "Other"))
#changing reference level of factor:
female$race_cat2 <- relevel(female$race_cat2 , ref = "White") 
#创建值标签
gender <- factor(data$gender, levels = c(1,2),          #名为gender的变量， 其中1表示男性，2表示女性
                 labels = c("male","female"))           #levels代表变量实际值，labels表示包含了理想值标签的字符型向量

#因子factor:分类变量和等级变量
f1 <- c("Male","Female")
f2 <- c("A","B","C","D", ordered=TRUE)                  #表示有序型变量,默认按照字母顺序
f3 <- factor(status, ordered = TRUE,                    
             levels = c("A","B","C","D"))               #按照指定顺序排序

#建立矩阵
mat_b <- matrix(1:8, ncol = 2 , nrow = 4)               #按列填充
mat_b[ , 1] <- c(2 , 3 , 5 , 6)
mat_a <- matrix(1:8, ncol = 2 , nrow = 4, byrow = TRUE, #按行填充
                dimnames = list(rnames,cnames))         #指定行列名

#建立数组array(类似于三维矩阵)
myarray <- array(1:24,c(2,3,4))

#建立list
L1 <- list(V1 = vector1 , T1 = data.frame2 , M1 = matrix3)
rm(V1, T1, M1)   #can remove the original objects since they are now saved in a list
length(L1)       #length is the number of objects in list
#Check/change item names
length(L1)
names(L1) <- c("Vector" , "Table" , "Matrix")
#extract an item of a list
L1[[1]] #the first element in list
L1[["T1"]]
L1$M1
#extract the 4th value of the vector V1 in L1
L1[[1]][4]
L1$V1[4]
L1[["V1"]][4]
#Flatten Lists: simplifies it to produce a vector which contains all the atomic components
unlist(x, recursive = TRUE, use.names = TRUE)

#Dataframe
dataf <- data.frame(a,b)
dim(dataf)  #dimensions of dataf
nrow(dataf) #number of rows
ncol(dataf) #number of cols
names(dataf)#names of cols
m1.t=as.data.frame(m1)
#创建一个可以输入数据的dataframe
mydata <- data.frame(age=numeric(0), gender=character(0), weight=numeric(0))
mydata <- edit(mydata)
#提取数据集的1，6，25列形成新的数据集
dataf.sub1 <- daraf[,c(1, 6, 25)]
dataf.sub2 <- dataf[,c("age","sex","IQ_performance")]          #用列名提取列
dataf.sub2 <- subset(dataf, select = c(age, sex, IQ_performance))
#Create a new variable based on data and conditions. 
wais.sub1$GenderLabel = ifelse(wais.sub1$Gender == 1, "Male", "Female")  #Male=1, Female=2
wais.sub1$math2=ifelse(wais.sub1$MathScore <= 10, 0, ifelse(wais.sub1$MathScore <=12, 1, 2))
#只提取满足某条件的列
wais.f = wais.sub1[wais.sub1$Gender == 2, ]
wais.f = wais.sub1[wais.sub1$GenderLabel == "Female", ]
wais.130 = wais.sub1[wais.sub1$IQperf > 130, ]

#the dimensions of the dataset
dim(survey) 
#removing incomplete cases删除缺失值
data_a <- na.omit(data)

#the split function
temp <- split(lab8a$data , f = lab8a$set)  #split(x,f)  x is data, f is factor splitting on

#在数据框内找数据
attach(dataset1)                   #attach(数据框)之后就进入了这个数据框，直接在里面找变量
...                                #其他运算
detach(dataset1)
with(ddataset, {                   #用with()实现在数据框内找数据，仅在括号内生效
  testid <- id                     #如果要创建在with()结构以外存在的对象，使用特殊赋值符<<-替代标准赋值符（<-）
})

#Generate random number (random objects can be reproduced)
set.seed(50)
rnorm(50)                          #之后再用rnorm生成随机数的时候生成相同的

#Create continuous uniform distributions
dataset = runif(n, min=1, max=5)   #default: lower and upper bounds are 0 and 1 respectively
#Sample from one distribution
s1=sample(dataset, n, replace=T)

#改变列名
colnames(acc_mean) <- c("Round","Agent","Accuracy")
colnames(dataf)[3] = "Vocabulary" #只改某列的名字
colnames(dataf)[4:5] = c("MathScore", "IQperf")
#改变行名
rownames(q4_mean2) <- c("A", "B", "C",'D')

#提取出id为X的某一行
idX <- dataframe[which(dataframe$Unique.ID == X),]

#根据A和B分类计算Y的平均值
Q <- aggregate(data_Y,list(A,B),mean)
#计算分组均值
Q <- tapply(dataframe1$Y,list(dataframe1$X1,dataframe1$X2),mean)
colMeans(m1) #各列平均值
#finds the mean cholesterol for each gender
tapply(chol2$chol, chol2$sex2, mean)
#arrange cho12 order by id
chol2 <- arrange(chol2 , id) 

#添加元素到向量
a=c(a, 5, 20) #adds 5 and 20 to existing values

#向量转化为矩阵
m1=matrix(a, ncol=2, byrow=F)  #read sth by vol竖着写入
m2=matrix(a, ncol=2, byrow=T)  #read sth by row横着写入

Mq <- cbind(a,b) #合并两列
M2 <- cbind(a,b) #合并两行
dataframe1 <- data.frame(v1,v2,v3) #几个向量合并为dataframe

#一个元素X重复N次
v <- rep(X,N)

#Sort (or order) a vector or factor (partially) into ascending or descending order
tmeans.sorted <- sort(tmeans)

##Loop
# for loop
for (i in seq(from=2, to=53, by=2)){
  a <- i+1
  for (m in 1:36){
    b <- m+1
  } 
}

# if loop判断
if (par_coop==1 & agent_coop==1){
  points[m] <- points[m] + 5
} else if (par_coop==1 & agent_coop==0){
  points[m] <- points[m] + 2
} else if(par_coop==0 & agent_coop==1){
  points[m] <- points[m] + 7
} else if (par_coop==0 & agent_coop==0){
  points[m] <- points[m] + 4
}

if(a < 0) {
  print("The value in a is smaller than 0")
} else {
  print("The value in a is not smaller than 0")
}

if (expectation<0 & agent_coop==0 | expectation>=0 & agent_coop==1){ #有两个条件和
  acc2[m,a] <- 1
}

#while
counter <- 0 #initialize the counter
while(counter < 5) {
  print(counter)
  counter <- counter + 1
}


## Statistics
describe(x)     #获得n,meam,sd,median,trimmed,mad,min,max,range,skew,kurtosis,se
idealfIQR(x)    #interquartile range
idealf(x)       #The Lower ideal 4th (q1) and upper ideal 4th (q2)
abs()           #绝对值

#截距
lsfit(x,y)$coef #x,y是两个向量
#线性回归
X=c(500,530,590,660,610,700,570,640)
Y=c(2.3, 3.1, 2.6, 3.0, 2.4, 3.3, 2.6, 3.5)
lsfit(X,Y)$coef
summary(lm(Y~X))$r.squared

#mean
mean(a)
tmean(a)        #default: 20% trimmed
median(a)
#sum求和
sum(a,b,c)

#数据中心化：数据集中的各项数据减去数据集的均值
scale(data, center=T,scale=F)
#数据标准化：数据集中的各项数据减去数据集的均值再除以数据集的标准差
scale(data, center=T,scale=T)

#Transfer to z score (of the normal distribution, defalt: mean = 0, sd = 1)
pnorm(x,mean,sd)
pnorm(x, 5, 2) #the probability p(X ≤ x) is 0.7854327.
qnorm(x) #the inverse of pnorm
qnorm(.5)# What is the Z-score of the 50th quantile of the normal distribution?
qnorm(pnorm(x, 5, 2), 0, 1) #the standard normal quantile corresponding to the probability obtained in (b) is 0.7906733
#Generate a vector of normally distributed random numbers
rnorm(n, mean = 70, sd = 5) #argument n is the number of numbers you want to generate
#skewness
data_skew <- skewness(x)
if (data_skew[1] > 0){
  print("The data is skewed positively")
} else if (data_skew[1] <0){
  print("The data is skewed negatively")
} else {
  print("The data is not skewed")
}

#Binomial Distribution
dbinom(x, size, p)  #x = number of successes, size = number of trials/observations, p = probability of success on each trial

#Check normality:
#(1). Use describe() to check the mean and median as well as skewness and kurtosis.
describe(female.res)
#(2). Use multi.hist(), boxplot(), and qqnorm() 
multi.hist(female.res)   #gives histogram, density plot, and normal curve
boxplot(female.res)
qqnorm(female.res)   #should be a line at 45 degree angle. 
qqline(female.res)
#(3). Use the Shapiro-Wilk test
shapiro.test(female.res)   #fail to reject means it's normal (in theory!)

#C.I.
qt(0.95,1) #Compute the critical quantile for 95% CI when sigma is unknown with df = 1 using qt()
qnorm(0.95) #compute the critical quantile for 95% CI when sigma is known, using qnorm()

#t-test
t.test(A, B, paired = TRUE, alternative = "two.sided")
#Paired-sampel t test
t.test(s2_mimic_offer, s2_nonmimic_offer, paired = TRUE, alternative = "two.sided")
wilcox.test(s2_mimic_offer, s2_nonmimic_offer, paired = TRUE, alternative = "two.sided")

#混合ANOVA
res.aov <- anova_test(data = data, dv = offer, wid = id, between = sex, within = mimicry)
get_anova_table(res.aov)

#Correlation
cor.test(offer,similarity) #person correlation

#Regression
lm.f <- lm(fev ~ height , data = female)
summary(lm.f)                      #将显示分析结果的统计概要
abline(lm.f)                       #plot the regression line
plot(lm.f)                         #将生成回归诊断图形
predict(lm.f, mynewdata)           #预测
#Get a confidence interval for the regression line
confint(object, level=0.95)
female.res <- residuals(lm.fc)     #Computing residuals
female.pred <- predict(lm.fc)      #Computing predicted values
#识别异常点
identify(x = cholm$wt , y = cholm$sbp)
#using robust method ols #ols(x, y, xout=FALSE, outfun=out, plotit=TRUE): default settings
ols(x = cholm$wt , y = cholm$sbp , xout = TRUE , plotit = TRUE)
#Quantile Regression: susceptible to leverage points, but guards against outliers on Y
rqfit(lab10b$x , lab10b$y , xout = T , res = F)  

#check for outliers for each variable individually.
out(lab10c$x)    #Checking leverage points
out(lab10c$y)    #Checking y outliers.

#least squares regression: The least squares estimate of β in the model
lsfit(lab10hw2$iq,lab10hw2$score)
lsfit(lab10hw2$iq,lab10hw2$score)$coef

#Wilcoxon-Mann-Whitney
wilcox.test(group1,group2)
wmw(group1,group2)
#Cliff
cidv2(group1,group2,plotit = FALSE)
#Brunner-Munzel
bmp(group1,group2,alpha = 0.5)
#Kolmogorov_Smirnow test
ks(group1,group2,sig = TRUE, alpha = 0.05)

## Draw Table
library(gridExtra)
grid.table(dataframe1)

## Draw graph
par(mfrow=c(1,3))    #multiple graphs in one window 在一个plot中1行画3个图
par(mfrow=c(1,1))    #returns plot to 1 panel

#histogram
hist(dataf$AngleDegs,breaks = 30,xlab = "XXX",col = "darkorchid")
multi.hist(IQ_f)     #Histogram, Density and Normal Fit在一个图上

ggplot(HW6_pt1_results, aes(x=HW6_pt1_results$AngleDegs)) +
  geom_histogram(aes(y = stat(count / sum(count))), color="black", fill="darkorchid")+
  xlab("x axis name") + 
  ylab("y axis name")+
  scale_x_continuous(breaks=seq(-25,65,10)) #the range of x axis

ggplot(Q7_data, aes(x = Q7_data$receivedmny2,y = (Q7_data$Emotion),fill = Q7_data$Emotion_label))+   #堆积柱状图
  geom_bar(stat ="identity",width = 0.6,position ="stack")+
  xlab("Money Received") + ylab("Accumulated AU Value")+
  theme(axis.line = element_line(size=0.5, colour = "black"))+
  scale_x_continuous(breaks=seq(0,10,1))+
  scale_fill_manual(name="AU-based Emotion",
                    values=c("#CC0033", "#3669BD", "#BCDA6A","#E9AD60", "#5BB5E3", "#8D86C5"))+
  theme(axis.text.y = element_blank())

ggplot(data=result_4_5, mapping=aes(x=result_4_5$money_receive,fill=result_4_5$feel))+     #上限一致的堆积柱状图
  geom_bar(stat="count",width=0.5,position='fill')+
  scale_fill_manual(name="Self-reported Feelings",values=c("#CC0033", "#3669BD", "#E9AD60", "#BCDA6A", "#5BB5E3", "#8D86C5"))+
  xlab("Money Received") + ylab("Count of Feelings")+
  scale_x_continuous(breaks=seq(0,10,1))+
  theme(axis.line = element_line(size=0.5, colour = "black"))

#line chart
ggplot(q4_reorganized_data,aes(x=Round,y=Cooperation_Rate,group=Condition))+
  geom_line(aes(linetype=Condition,color=Condition),linetype="solid")+
  scale_x_continuous(breaks=seq(1,13,1))+
  scale_y_continuous(breaks=seq(0,1,0.05))

ggplot(Q1,aes(x=Group.2,y=x,colour=Group.1,group=Group.1))+ #line chart分组画2条线
  geom_line()+
  geom_point()+
  ylim(0,10)+
  xlab("Mimicry") + ylab("Money Sent to Agent ($)")+
  theme(legend.title=element_blank())+
  theme(axis.line = element_line(size=0.5, colour = "black"))

#error bar
Q_SE <- summarySE(f2_data, measurevar="offer", groupvars="similarity")
f2<- ggplot(Q,aes(x=similarity,y=offer))+
  geom_line()+
  geom_point()
f2 + geom_errorbar(aes(ymin=offer-Q_SE$se, ymax=offer+Q_SE$se), width=.1)

##Write my own function
my_mean = function(x){     #x是输入input
  n <- length(x)
  mean <- sum(x)/n
  return(mean)             #return()内的是function输出结果
}
my_mean(a)

## R markdown
#空白行
&nbsp;  
#加粗
**Qianhui Ni**
#把向量放进句子里
My SVO score is `r my_svo_score`, and my category is `r my_svo_categ`.
#insert pictures
![Caption for the picture.](q7a_ext.png) 