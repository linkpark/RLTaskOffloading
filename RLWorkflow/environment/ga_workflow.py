import array
import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

def basic_ga(task_graph, eval_one_max_func, cxpb=0.2, mutpb=0.08, ngen=30, npop=50):  # only local or remote, i.e., 0 or 1.
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, task_graph.task_number)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        individual = numpy.array(individual)
        return -eval_one_max_func(individual, task_graph),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(64)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=False)

    return pop, log, hof


# if __name__ == "__main__":
#     x_graph = Task
#     _, _, hof = basic_ga(x_graph, cxpb=0.5, mutpb=0.1, ngen=100, npop=200)
#     print(hof[0])