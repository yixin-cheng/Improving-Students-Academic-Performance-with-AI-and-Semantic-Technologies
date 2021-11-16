"""
This file is for the feature selection based on Genetic Algorithm and SVM
"""
# import required libraries
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate


# ## Step 2: Define settings
# 1. DNA size: the number of bits in DNA
# 2. Population size
# 3. Crossover rate
# 4. Mutation rate
# 5. Number of generations
class Selector:
    def __init__(self,path,target_path):
        self.target_path=target_path
        self.path=path
        self.df=pd.read_csv(self.path)
    # define GA settings
        self.DNA_SIZE = len(self.df.columns)-3  # number of bits in DNA which equals the number of features(exclude the grade)
        self.POP_SIZE = 1000  # population size
        self.CROSS_RATE = 0.75  # DNA crossover probability
        self.MUTATION_RATE = 0.002  # mutation probability
        self.N_GENERATIONS = 100  # generation size
        self.evolution()

    # ## Step 3: Define fitness, select, crossover, mutate functions

    def get_fitness(self,pop,path):
        """
        This function calculates the fitness (accuracy) in each DNA based on the Support Vector Machine algorithm
        :param pop: population
        :param path: the path is to the preprocessed data set
        :return: a list of accuracy of each DNA
        """
        res = []
        for element in pop:
            data = pd.read_csv(path, header=None, index_col=0)
            data.drop(data.columns[0],axis=1)
            droplist = []
            for i in range(len(element)):
                if element[i] == 0:
                    droplist.append(i)
            data=data.drop(data.columns[droplist], axis=1)
            print('Individual: ',element)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            scoring = {'accuracy': 'accuracy',
                       'precision': 'precision',
                       'recall': 'recall',
                       'f1': 'f1'}
            # SVM for fitness computation
            svc = SVC(C=0.4)
            scores = cross_validate(svc, X, y, scoring=scoring, cv=5) #peform 5-fold validaiton
            res_tem = {"Acc": np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
                       "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
            # res['SVC'] = res_tem
            print('Result: ',res_tem)
            res.append(res_tem["Acc"])
        return res


    # define population select function based on fitness value
    # population with higher fitness value has higher chance to be selected
    def select(self,pop, fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True,
                           p=fitness / sum(fitness))
        return pop[idx]


    # define gene crossover function
    def crossover(self,parent, pop):
        if np.random.rand() < self.CROSS_RATE:
            # randomly select another individual from population
            i = np.random.randint(0, self.POP_SIZE, size=1)
            # choose crossover points(bits)
            cross_points = np.random.randint(0, 2, size=self.DNA_SIZE).astype(np.bool)
            # produce one child
            parent[cross_points] = pop[i, cross_points]
        return parent


    # define mutation function
    def mutate(self,child):
        for point in range(self.DNA_SIZE):
            a = np.random.rand()
            if a < self.MUTATION_RATE:
                # print(a)
                child[point] = 1 if child[point] == 0 else 0
        return child


    # ## Step 4: Start training GA
    # 1. randomly initialise population
    # 2. determine fitness of population
    # 3. repeat
    #     1. select parents from population
    #     2. perform crossover on parents creating population
    #     3. perform mutation of population

    def evolution(self):
        """
        the whole process of genetic algorithm
        """
        # initialise population DNA
        pop = np.random.randint(0, 2, (self.POP_SIZE, self.DNA_SIZE))
        for t in range(self.N_GENERATIONS):
            # train GA
            # calculate fitness value
            fitness = self.get_fitness(pop,self.path)  # translate each NDA into accuracy which is fitness
            # if the generation reaches the max, then abandon the bad performance feature and save the rest of features to a new file
            if t == self.N_GENERATIONS - 1:
                res = pop[np.argmax(fitness), :]
                print("Most fitted DNA: ", pop[np.argmax(fitness), :])
                data=pd.read_csv(self.path, header=None, index_col=0)
                data.drop(data.columns[0], axis=1)
                droplist = []
                for i in range(len(res)):
                    if res[i] == 0:
                        droplist.append(i)
                print("Abandoned feature index: ", droplist)
                data=data.drop(data.columns[droplist], axis=1)
                if type(data) is not None:
                    data.to_csv(self.target_path, header=None,index=False)
            # select better population as parent 1
            pop = self.select(pop, fitness)
            # make another copy as parent 2
            pop_copy = pop.copy()

            for parent in pop:
                # produce a child by crossover operation
                child = self.crossover(parent, pop_copy)
                # mutate child
                child = self.mutate(child)
                # replace parent with its child
                parent[:] = child
