# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd
import random

experiment_name = 'Results_task1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

df = pd.DataFrame(columns=["Enemy", "Algorithm", "Run", "Repetition", "Energy enemy", "Enegy player"])
index_df = 0
for EA_algorithm in ['Roulette','Tournament']:
    for enemy_name in range(1,4):
        for run_nummer in range(1,11):
                   
            # Load solution and specify values
            bsol = np.loadtxt(experiment_name+'/Best_individuals_'+str(EA_algorithm)+'_e'+str(enemy_name)+'_i'+str(run_nummer)+'.txt') # txt file with best solution for 1 enemy with 1 EA for a specific run
            #print(bsol)
            enemy_nr = enemy_name # emeny where the solution is created for
            EA_name = EA_algorithm # EA that is used to create solution
            run_nr = run_nummer # 1 to 10
            
            # Environment
            n_hidden_neurons = 5
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy_nr],
                              playermode="ai",
                              player_controller=player_controller(n_hidden_neurons),
                              enemymode="static",
                              level=2,
                              speed="fastest")
            
            # run 5 times and add to df
            for i in range(0,5):
                random.seed(i)
                f, p, e, t = env.play(pcont=bsol)
                df.loc[index_df, ] = [enemy_nr, EA_name, run_nr, i+1, e, p]
                index_df = index_df + 1
    
# write data to file
print(df)
df.to_csv("df_boxplots_roulette_tournament_new.csv")