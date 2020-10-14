from __future__ import division
import random

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence 
import sys, os
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
import numpy as np
import json

experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

player_life = 100
enemy_life = 100
dom_u = 1
dom_l = -1
n_population = 5 #100
gens = 80 #120
mate = 1
#mutation = 0.2
mutations = [0.9]#[0.2,0.3,0.5,0.7,0.9]
last_best = 0
n_hidden_neurons = 10 #number of possible actions
budget = 1500
#enemies = 3
runs = 10
envs = []
eatype = "Roulette"
# initializes environment with ai player using random controller, playing against static enemy

enemies_g_1 = [7,8]
enemies_g_2 = [2,5,6]


env_1 = Environment(experiment_name=experiment_name,
                        enemies=enemies_g_1,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        timeexpire = budget)

env_2 = Environment(experiment_name=experiment_name,
                        enemies=enemies_g_2,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        timeexpire = budget)

envs.append(env_1)
envs.append(env_2)

env = envs[0]
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

creator.create('FitnessBest', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessBest, player_life = player_life, enemy_life = enemy_life)

tb = base.Toolbox() # contains the evolutionary operators
tb.register('indices', np.random.uniform, dom_l,dom_u) 
# initRepeat: Call the function container (creator.Individual) with a generator function (tb.indiceis) corresponding to the calling n (weights) times the function func.
tb.register('individual', tools.initRepeat, creator.Individual, tb.indices, n = n_weights)
tb.register('population', tools.initRepeat, list, tb.individual, n = n_population)


def simulation(env,x):
    fitness,p_l,e_l,time_ = env.play(pcont=x)
    return fitness

def evaluate(x,env):
    tmp = simulation(env, x) + 10
    return (tmp,)

def evalpop(pop,env):
    to_evaluate_ind = [ind for ind in pop if not ind.fitness.valid]
    tmp = [env for i in range(len(to_evaluate_ind))]
    fits = tb.map(tb.evaluate,to_evaluate_ind,tmp)
    
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    return (pop)

def rec(pop):
    fits = [ind.fitness.values[0] for ind in pop]
    record = {"mean": sum(fits)/len(pop), "max":np.max(fits)}
    return record

def writestats(x,name):
    f=open(name,"w+")
    for item in x:
        f.write(json.dumps(item)+"\n")

def storestats(pop,g,r,e,hof,mut):
    hof.update(pop)
    record = rec(pop)
    log.record(enemy=e, run=r, gen=g, individuals=len(pop), mut=mut, **record)
    
def writebest(inds,eatype,mutation,enemy):
    for count,ind in enumerate(inds):
        tmp = count + 1
        f=open(eatype+"_e" +str(enemy)+ "_i" +str(tmp) +".txt","w+")
        for weight in ind:
            f.write(str(weight)+"\n")
        
def mutUniform(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a float uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from which to draw the new
                value
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from which to draw the new
               value
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.uniform(xl, xu)

    return individual,
    
tb.register("evaluate", evaluate)
tb.register("mate", tools.cxTwoPoint) # crossover operator 
tb.register("mutate", mutUniform, low = dom_l, up = dom_u, indpb = 0.1)
tb.register("select", tools.selRoulette)

record = {}
log = tools.Logbook()
log.header = ['enemy','run','gen','individuals','mut', 'mean','max']
def main():
    for mutation in mutations:
        for count, env in enumerate(envs):
            bestinds = []
            for run in range(0,runs):
                pop = tb.population(n=n_population)
                pop = evalpop(pop,env)
                hof = tools.HallOfFame(1, similar=np.array_equal)
                storestats(pop,0,run+1,count+1,hof,mutation)
                for g in range(1,gens):
                    pop = tb.select(pop,len(pop))
                    offs = algorithms.varAnd(pop,tb,mate,mutation)
                    offs = evalpop(offs,env)
                    pop = offs
                    storestats(pop,g,run+1,count+1,hof,mutation)
                bestinds.append(hof[0])
            writebest(bestinds,"Best_individuals_"+eatype,mutation,count+1)
        writestats(log,"log_stats_"+eatype+".txt")    
        
if __name__ == "__main__":
    main()
    