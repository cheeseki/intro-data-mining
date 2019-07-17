filename_xiao = "C:\\Users\\xwunuke\\Desktop\\STATS final\\train_V2.csv"
#filename_chen = "/Users/chen.liang/Desktop/415_Project/train_V2.csv"
data = read.csv(file=filename_xiao, header=TRUE, sep=",")

####### clean data ###########
# delete irrelevant features
data_clean = data[complete.cases(data), ]
irrelevant_features = c('groupId', 'matchId', 'Id', 'rankPoints', 'maxPlace' )
## IDs' are not features
## rankPoints: the ranking is inconsistent in dataset
## maxPlace: basically the same as numGroups
data_clean[, irrelevant_features] = list(NULL)
## fpp: First Person Perspective
data_clean = data_clean[which(data_clean$matchType == 'duo' |
                              data_clean$matchType == 'duo-fpp' |
                              data_clean$matchType == 'solo' |
                              data_clean$matchType == 'solo-fpp' |
                              data_clean$matchType == 'squad' |
                              data_clean$matchType == 'squad-fpp'), ]
data_clean$is_duo = ifelse((data_clean$matchType=='duo-fpp' | data_clean$matchType=='duo'), 1, 0)
data_clean$is_solo = ifelse((data_clean$matchType=='solo-fpp' | data_clean$matchType=='solo'), 1, 0)
# squad is the baseline
#data_clean$is_squad = ifelse((data_clean$matchType=='squad-fpp' | data_clean$matchType=='squad'), 1, 0)
data_clean$is_fpp = ifelse((data_clean$matchType=='squad-fpp' |
                            data_clean$matchType=='duo-fpp' |
                            data_clean$matchType=='solo-fpp'), 1, 0)
data_clean$matchType = NULL
# code weapon: one can take at most two weapons.
data_clean$weaponsAcquired = ifelse(data_clean$weaponsAcquired < 2, 0, 1)
colnames(data_clean)[colnames(data_clean)=='weaponsAcquired'] = 'is_loaded'
#these variables are strongly influenced by time spend in the match
data_clean$damageDealt = data_clean$damageDealt/data_clean$matchDuration 
data_clean$rideDistance = data_clean$rideDistance/data_clean$matchDuration
data_clean$swimDistance = data_clean$swimDistance/data_clean$matchDuration
data_clean$walkDistance = data_clean$walkDistance/data_clean$matchDuration
# code target of prediction: define "winner" as ranking top 85th percentile
data_clean$is_Winner = ifelse(data_clean$winPlacePerc >= 0.85, 1, 0)
rm(data)

####### Feature Normalization ########
normalize <- function(x){
  range = max(x)-min(x)
  return((x-min(x))/range)
}

for (i in 1:(ncol(data_clean)-1)){
  data_clean[,i] = normalize(as.numeric(data_clean[,i]))
}

####### Feature Selection ########
data_featured = data_clean
data_featured$numGroups = NULL
data_featured$winPlacePerc = NULL

####### obtain train and test data ###########
set.seed(123)
#### to make life easier, we sample a subset ####
ind = sample(seq_len(nrow(data_featured)), size = 30000)
data_subset = data_featured[ind, ]
## create train and test sets
train_size = floor(0.5 * nrow(data_subset))
train_ind = sample(seq_len(nrow(data_subset)), size = train_size)
train = data_subset[train_ind, ]
test = data_subset[-train_ind, ]
rm(train_ind)

####### Logistic Regression #########
model = glm(train$is_Winner~., data=train,family=binomial(link=logit))
result_LG = predict(model, test[,1:(ncol(test)-1)])
result_LG = ifelse(result_LG>0.5, 1, 0)
sum(result_LG==test$is_Winner)/nrow(test)
summary(model)

####### fit SVM ##########
library(e1071) # for svm
# linear kernel ###########
### try different costs
x=vector()
cost_set = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50)
for(i in 1:length(cost_set)){
  model = svm(train$is_Winner~., data=train, kernel="linear", type="C-classification", cost=cost_set[i], scale=FALSE)
  result = predict(model, test[,1:(ncol(test)-1)])
  x[i]=sum(result==test$is_Winner)/nrow(test)
}
index = which(x==max(x))
best_cost = cost_set[index] 
best_cost

# radial kernel ########
### try different costs and gammas
x=vector()
cost_list = vector()
kernel_list = vector()
cost_set = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50)
kernel_set = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50)
for(i in 1:length(cost_set)){
  for(j in 1:length(kernel_set)){
    model = svm(train$is_Winner~., data=train, kernel="radial", type="C-classification", gamma = 1/(2*kernel_set[j]^2), cost=cost_set[i], scale=FALSE)
    result = predict(model, test[,1:(ncol(test)-1)])
    accuracy = sum(result==test$is_Winner)/nrow(test)
    x[(i-1)*length(cost_set)+j] = accuracy
    print(accuracy)
    cost_list[(i-1)*length(cost_set)+j] = cost_set[i]
    kernel_list [(i-1)*length(cost_set)+j] = kernel_set[j]
  }
}
index = which(x==max(x))
cost_list[index]
kernel_list[index]

# clean the environment
rm(result, result_LG, kernel_list, kernel_set, i, j, index, irrelevant_features, 
   cost_set, cost_list, best_cost, accuracy, model, ind)

####### Play with trees and forests ##########
library('rpart')
library('rpart.plot')

TreeFit<-rpart(as.factor(train$is_Winner)~.,data=train, method = "class", parms=list(split="GINI"))
rpart.plot(TreeFit,type=4,branch=1)
result_rpart = predict(TreeFit, test[,1:(ncol(test)-1)], type="c")
sum(result_rpart==test$is_Winner)/nrow(test)

TreeFit_2<-rpart(as.factor(train$is_Winner)~.,data=train, method = "class", parms=list(split="information"))
rpart.plot(TreeFit_2,type=4,branch=1)
result_rpart_2 = predict(TreeFit_2, test[,1:(ncol(test)-1)], type="c")
sum(result_rpart_2==test$is_Winner)/nrow(test)

library('randomForest')
rFM = randomForest(as.factor(train$is_Winner)~., data=train, importance = TRUE)
result_RF = predict(rFM, test[,1:(ncol(test)-1)])
sum(result_RF == test$is_Winner)/nrow(test)
graph_data = sort(rFM$importance[,3])
barplot(graph_data, las=2, ylim = c(0, 0.13),cex.names=0.6)
box()
#importance(rFM,type=1,scale=TRUE)[order(importance(rFM,type=1)[,1]),]

###### Finished #######
# an example of correlation between features
attach(data_clean)
rel_1 = lm(damageDealt ~ killPlace)
summary(rel_1)