import numpy as np
import pandas as pd

from deap import base
from deap import creator
from deap import tools

import random

from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import multiprocessing


def RfcParameters(icls):
    genome = list()
    # n_estimators
    genome.append(random.randint(50, 100))
    # max_depth
    genome.append(random.uniform(0.1,10))
    # criterion
    listCriterion = ['gini', 'entropy']
    genome.append(listCriterion[random.randint(0, 1)])
    # class_weight
    listClass = ['balanced', 'balanced_subsample']
    genome.append(listClass[random.randint(0, 1)])
    return icls(genome)
    
def RfcParametersFitness(y, df, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    # print(individual)
    estimator = RandomForestClassifier(n_estimators=(individual[0]), max_depth=individual[1], criterion=individual[2], class_weight=individual[3], random_state=0)
    
    resultSum = 0
    
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        # recall = tp/(tp + fn) 
        if tp + fp == 0:
            continue
        precision = tp/(tp + fp)
        # f1 = 2*precision*recall/(precision+recall)
        # f1 = tp/(tp + 0.5*(fp + fn))
        # B = 2
        # factor = (1+B*B)*tp
        # fB = factor/(factor + B*B*fn + fp)
        resultSum = resultSum + precision # recall # fB # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej

    return resultSum / split,

def mutationRfc(individual):
    numberParamer= random.randint(0, len(individual)-1)
    if numberParamer==0:
        # n_estimators        
        individual[0]=random.randint(50, 100)
    elif numberParamer == 1:
        # max_depth
        individual[1]=random.uniform(0.1,10)
    elif numberParamer == 2:
        # criterion
        listCriterion = ['gini', 'entropy']
        individual[2] = listCriterion[random.randint(0, 1)]
    elif numberParamer == 3:
        # class_weight
        listClass = ['balanced', 'balanced_subsample']
        individual[3] = listClass[random.randint(0, 1)]

def main():
    df = pd.read_csv('train.csv')

    data = df.to_numpy()

    reduced = data[:,0:-1]
    y_train = data[:,-1]


    sizePopulation = 50
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 100
    numberElitism = 1
    processes = 12

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    pool = multiprocessing.Pool(processes=processes)
    toolbox.register("map", pool.map)

    toolbox.register('individual', RfcParameters, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", RfcParametersFitness,y_train, reduced)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutationRfc)

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit    


    import time

    data = []

    g = 0
    start=time.time()
    while g < numberIteration:
        g = g+1
        print("\r", g, 'populacja', len(pop), end='')
        offspring = toolbox.select(pop, len(pop)-numberElitism)
        offspring = list(map(toolbox.clone, offspring))
        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring + listElitism
        fits = [ind.fitness.values[0] for ind in pop]

        row = []
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        row.append(min(fits))
        row.append(mean)
        row.append(std)
        data.append(row)
        best_ind = tools.selBest(pop, 1)[0]
        
    end = time.time()
    print("\nGen %s, Best individual is %s, %s in %s s" % (g, best_ind, best_ind.fitness.values,(end-start)))

    # data = np.array(data)
    # plt.plot(data[:,0], label='max')
    # plt.plot(data[:,1], label='mean')
    # plt.plot(data[:,2], label='std')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
