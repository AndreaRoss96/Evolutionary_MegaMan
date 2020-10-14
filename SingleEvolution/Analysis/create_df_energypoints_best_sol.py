# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:40:11 2020

@author: corien
"""
import sys, os
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd

experiment_name = 'solutions_EA1_enemy1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Load solution and specify values
bsol = np.loadtxt(experiment_name+'/Solution1.txt') # txt file with best solution for 1 enemy with 1 EA for a specific run
#print(bsol)
enemy_nr = 1 # emeny where the solution is created for
EA_name = "EA1" # EA that is used to create solution
run_nr = 1 # 1 to 10
df = pd.DataFrame(columns=["Enemy", "Algorithm", "Run", "Repetition", "Energy enemy", "Enegy player"])

# Environment
n_hidden_neurons = 10
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy_nr],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# run 5 times and add to df
for i in range(0,5):
    f, p, e, t = env.play(pcont=bsol)
    df.loc[i, ] = [enemy_nr, EA_name, run_nr, i+1, e, p]
    
# write data to file
print(df)
df.to_csv("df_boxplot.csv")