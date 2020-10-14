library(readr)
library(ggplot2)

df <- read_csv("Computational_science/Evolutionary_computing/evoman_framework-master/dataframe_boxplot.csv")

# Aggregate data

df$X1 = NULL
df$Repetition = NULL
agg <- aggregate(list(df$`Energy enemy`, df$`Enegy player`), by = list(df$Enemy, df$Algorithm, df$Run), mean)
enemy1 <- agg[agg[ ,1] == 1, ]
enemy2 <- agg[agg[ ,1] == 2, ]
enemy3 <- agg[agg[ ,1] == 3, ]
colnames(enemy1) <- c("Enemy","Algorithm","Run","Enegy_enemy","Energy_player")
colnames(enemy2) <- c("Enemy","Algorithm","Run","Enegy_enemy","Energy_player")
colnames(enemy3) <- c("Enemy","Algorithm","Run","Enegy_enemy","Energy_player")

enemy1_player <- enemy1
enemy1_player$Enegy_enemy = NULL
enemy1_player$Group <- c(rep("Player",20))
colnames(enemy1_player) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy1_enemy <- enemy1
enemy1_enemy$Energy_player <- NULL
enemy1_enemy$Group <- c(rep("Enemy",20))
colnames(enemy1_enemy) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy2_player <- enemy2
enemy2_player$Enegy_enemy = NULL
enemy2_player$Group <- c(rep("Player",20))
colnames(enemy2_player) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy2_enemy <- enemy2
enemy2_enemy$Energy_player <- NULL
enemy2_enemy$Group <- c(rep("Enemy",20))
colnames(enemy2_enemy) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy3_player <- enemy3
enemy3_player$Enegy_enemy = NULL
enemy3_player$Group <- c(rep("Player",20))
colnames(enemy3_player) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy3_enemy <- enemy3
enemy3_enemy$Energy_player <- NULL
enemy3_enemy$Group <- c(rep("Enemy",20))
colnames(enemy3_enemy) <- c("Enemy","Algorithm","Run","Enegy","Group")

enemy1 <- rbind(enemy1_player,enemy1_enemy)
enemy2 <- rbind(enemy2_player,enemy2_enemy )
enemy3 <- rbind(enemy3_player,enemy3_enemy )
                
# creeer boxplot enemy 1
enemy1_boxplot <- ggplot(enemy1, aes(x=Algorithm, y=Enegy, fill=Group)) + 
  xlab("Algorithm") + 
  ylab("Energy") +
  labs(fill = "") +
  ggtitle("Enemy 1") +
  geom_boxplot()

enemy1_boxplot

# creeer boxplot enemy 2
enemy2_boxplot <- ggplot(enemy2, aes(x=Algorithm, y=Enegy, fill=Group)) + 
  xlab("Algorithm") + 
  ylab("Energy") +
  labs(fill = "") +
  ggtitle("Enemy 2") +
  geom_boxplot()

enemy2_boxplot

# creeer boxplot enemy 3
enemy3_boxplot <- ggplot(enemy3, aes(x=Algorithm, y=Enegy, fill=Group)) + 
  xlab("Algorithm") + 
  ylab("Energy") +
  labs(fill = "") +
  ggtitle("Enemy 3") +
  geom_boxplot()

enemy3_boxplot