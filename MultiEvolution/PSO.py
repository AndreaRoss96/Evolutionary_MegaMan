import sys, os
sys.path.insert(0, 'evoman') 
#proj
from evoman.environment import Environment
from demo_controller import player_controller
import json
#std lib
import operator
import random
#math lib
import numpy as np
import math
# deap
from deap import base,benchmarks,creator,tools

experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

player_life = 100
enemy_life = 100
dom_u = 1
dom_l = -1
n_population = 5 #100
gens = 80
mate = 1
last_best = 0
n_hidden_neurons = 10 #number of possible actions
budget = 1500
runs = 10
envs = []
eatype = "PSO"
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

def rec(pop):
    fits = [ind.fitness.values[0] for ind in pop]
    record = {"mean": sum(fits)/len(pop), "max":np.max(fits)}
    return record

def writestats(x,name):
    f=open(name,"w+")
    for item in x:
        f.write(json.dumps(item)+"\n")

def storestats(pop,g,r,e,hof):
    hof.update(pop)
    record = rec(pop)
    log.record(enemy=e, run=r, gen=g, individuals=len(pop), **record)

def writebest(inds,eatype,enemy):
    for count,ind in enumerate(inds):
        tmp = count + 1
        f=open(eatype+"_e" +str(enemy)+ "_i" +str(tmp) +".txt","w+")
        for weight in ind:
            f.write(str(weight)+"\n")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, player_life = player_life, enemy_life = enemy_life, speed=list, 
    pmin=None, pmax=None, smin=None, smax=None, best=None) # speed limits set to None, they are setted later

def generate(size, pmin, pmax, smin, smax):
    '''
    Creates a particle and initializes its attributes,
    except for the attribute best, which will be set only after evaluation.
    '''
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part

def updateParticle(part, best, phi1, phi2):
    '''
    Computes the speed, then limits the speed values between smin and smax, 
    and finally computes the new particle position.
    '''
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = math.copysign(part.smin, speed) # copysign --> returns the value of the first param and the sign of the second param
        elif speed > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    for i, val in enumerate(part):
        if val > part.pmax:
            part[i] = part.pmax
        elif val < part.pmin:
            part[i] = part.pmin

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

tb = base.Toolbox()
tb.register("particle", generate, size=n_weights, pmin=dom_l, pmax=dom_u, smin=-0.10, smax=0.10)
tb.register("population", tools.initRepeat, list, tb.particle)
tb.register("update", updateParticle, phi1=0.5, phi2=0.5)
tb.register("evaluate", evaluate)

log = tools.Logbook()

def main():
    for count, env in enumerate(envs):
        bestinds = []
        for run in range(runs):
            pop = tb.population(n=n_population)
            pop = evalpop(pop,env)
            hof = tools.HallOfFame(1, similar=np.array_equal)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            log.header = ['enemy','run','gen','particle'] + stats.fields

            storestats(pop,0,run+1,count+1,hof)
            best = None # contains the best particle ever fonud

            for g in range(gens):
                for part in pop:
                    part.fitness.values = tb.evaluate(part,env)
                    if part.best is None or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if best is None or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values
                for part in pop:
                    tb.update(part, best)
                storestats(pop,g,run+1,count+1,hof)

                print(log.stream)
            
            bestinds.append(hof[0])
        writebest(bestinds,"Best_individuals_"+eatype,count+1)
    writestats(log,"log_stats_"+eatype+".txt")

if __name__ == "__main__":
    main()

