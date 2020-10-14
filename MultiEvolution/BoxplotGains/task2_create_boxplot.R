library(readr)
library(ggplot2)

df <- read_csv("Computational_science/Evolutionary_computing/evoman_framework-master/df_task2_good.csv")

# Creating boxplot
df$X1 = NULL
df$Repetition = NULL
df$gain <- df$`Enegy player` - df$`Energy enemy`

agg <- aggregate(list(df$`Energy enemy`, df$`Enegy player`, df$gain), by = list(df$Enemy, df$Algorithm, df$Run, df$`Enemy training group`), mean)
colnames(agg) <- c("Enemy","Algorithm","Run","Enemy_training_group","Energy_enemy","Energy_player","Gain")
new_column <- c()
for (row in agg$Energy_player){
  if (row==0){
    win = 0
  }
  else{
    win = 1  
  }
  new_column <- append(new_column, win)
}

agg$Win <- new_column
agg2 <- agg
agg2$Enemy <- NULL
agg3 <- aggregate(list(agg2$Energy_enemy, agg2$Energy_player, agg2$Gain, agg2$Win), by = list(agg2$Algorithm, agg2$Run, agg2$Enemy_training_group), sum)
colnames(agg3) <- c("Algorithm","Run","Enemy_training_group","Energy_enemy","Energy_player","Gain","Nr defeated enemies")
agg3$AlgorithmTrainGroup <- paste(agg3$Algorithm, agg3$Enemy_training_group)
agg3$Enemy_training_group <- factor(agg3$Enemy_training_group)

ggplot(agg3, aes(x=Enemy_training_group, y=Gain, fill=Algorithm)) + 
  ggtitle("Comparison of gain by EA and training group") +
  xlab("Training group of enemies") +
  scale_fill_discrete(name = "Algorithm", labels = c("PSO", "GA")) +
  scale_x_discrete(labels=c("1" = "[7,8]", "2" = "[2,5,6]")) +
  geom_boxplot()

# Statistical testing differences EA and training group
# Normal distributed
shapiro.test(agg3[agg3[, 8]=="PSO 1", 6])
shapiro.test(agg3[agg3[, 8]=="Roulette 1", 6])
shapiro.test(agg3[agg3[, 8]=="PSO 2", 6])
shapiro.test(agg3[agg3[, 8]=="Roulette 2", 6])

# Equal variance
library(car)
leveneTest(Gain ~ AlgorithmTrainGroup, data = agg3)

# One way anova
res.aov <- aov(Gain ~ AlgorithmTrainGroup, data = agg3)
summary(res.aov)

# create table with mean +std for comparison baseline paper
PSO1 <- agg3[agg3[, 8]=="PSO 1", 6]
Roulette1 <- agg3[agg3[, 8]=="Roulette 1", 6]
PSO2 <- agg3[agg3[, 8]=="PSO 2", 6]
Roulette2 <- agg3[agg3[, 8]=="Roulette 2", 6]
print(paste("Mean PSO1",mean(PSO1)))
print(paste("Mean PSO2",mean(PSO2)))
print(paste("Mean GA1",mean(Roulette1)))
print(paste("Mean GA2",mean(Roulette2)))
print(paste("std PSO1",sd(PSO1)))
print(paste("std PSO2",sd(PSO2)))
print(paste("std GA1",sd(Roulette1)))
print(paste("std GA2",sd(Roulette2)))