import math
import random
from matplotlib import pyplot as plt
import numpy as np
import time


class Anthill:
    ANT_MODE = 'ANT_MODE'
    MIN_MAX_MODE = 'MIN_MAX_MODE'

    class Edge:
        def __init__(self, node_1, node_2, weight, initial_pheromone):
            self.node_1 = node_1
            self.node_2 = node_2
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def choose_node(self):
            unvisited_nodes = [node for node in range(
                self.num_nodes) if node not in self.tour]
            heuristic = pheromone = 0.0
            probability_nodes = [0.0] * len(unvisited_nodes)

            for unvisited_node in unvisited_nodes:
                heuristic += pow(1 / self.edges[self.tour[-1]]
                                 [unvisited_node].weight, self.beta)
                pheromone += pow(self.edges[self.tour[-1]]
                                 [unvisited_node].pheromone, self.alpha)
            for i in range(len(unvisited_nodes)):
                probability_nodes[i] = (pow(self.edges[self.tour[-1]][unvisited_nodes[i]].pheromone, self.alpha) * pow(
                    1 / self.edges[self.tour[-1]][unvisited_nodes[i]].weight, self.beta)) / (heuristic * pheromone)
            node_index = probability_nodes.index(max(probability_nodes))

            return unvisited_nodes[node_index]

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self.choose_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]
                                            ][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode, colony_size=10, min_param=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_param=1.0, initial_pheromone=1.0, steps=100, nodes=None):
        self.mode = mode
        self.colony_size = colony_size
        self.min_param = min_param
        self.rho = rho
        self.pheromone_param = pheromone_param
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                    initial_pheromone)

        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges)
                     for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def add_pheromone(self, tour, distance):
        pheromone_to_add = self.pheromone_param / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]
                                ].pheromone += pheromone_to_add

    def ant_process(self):
        for step in range(self.steps):
            for ant in self.ants:
                self.add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance
            if float(step + 1) / float(self.steps) <= 0.75:
                self.add_pheromone(iteration_best_tour,
                                   iteration_best_distance)
                max_pheromone = self.pheromone_param / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self.add_pheromone(self.global_best_tour,
                                   self.global_best_distance)
                max_pheromone = self.pheromone_param / self.global_best_distance
            min_pheromone = max_pheromone * self.min_param
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

    def run(self):
        if self.mode == self.ANT_MODE:
            self.ant_process()
        elif self.mode == self.MIN_MAX_MODE:
            self.max_min()
        # print(
        #     'Route : {0}'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        # print('Distance : {0}\n'.format(
        #     round(self.global_best_distance, 2)))

        return self.global_best_distance

    def plot(self, annotation_size=8, dpi=120, iteration=0):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=1)
        plt.scatter(x, y, s=math.pi * 4)
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)

        name = 'plots/{0}_tour_{1}.png'.format(self.mode, iteration)
        plt.savefig(name, dpi=dpi)
        plt.gcf().clear()


def statistic_plot(ant_distances, min_max_distances, iterations):
    plt.plot(list(range(1, iterations + 1)), ant_distances)
    plt.plot(list(range(1, iterations + 1)), min_max_distances)

    plt.legend(['ant', 'min_max'], loc='upper left')
    name = 'plots/statistic.png'
    plt.savefig(name, dpi=120)
    plt.show()
    plt.gcf().clear()


if __name__ == '__main__':
    colony_size = 50
    steps = 50
    nodes_count = 50

    ant_distances = []
    min_max_distances = []
    ant_times = []
    min_max_times = []

    iteratios = 20
    config = [
        {'steps': 50, 'nodes': 50, 'colony_size': 50, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 50, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 100, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 0.5,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 2.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 0.5, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 2.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 0.5, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 2.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 0.1,
            'beta': 0.1, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 2.0,
            'beta': 2.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 5.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 2.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 2.0,
            'beta': 1.0, 'pheromone_param': 2.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 2.0, 'pheromone_param': 0.5, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 1.0,
            'beta': 2.0, 'pheromone_param': 2.0, 'rho': 0.1},
        {'steps': 100, 'nodes': 50, 'colony_size': 200, 'alpha': 0.1,
            'beta': 0.1, 'pheromone_param': 2.0, 'rho': 0.1},
    ]
    nodes = [(random.uniform(-1500, 1500), random.uniform(-1500, 1500))
             for _ in range(0, nodes_count)]
    for i in range(iteratios):
        start_time_ant = time.time()
        # mode, colony_size=10, min_param=0.001, alpha=1.0, beta=3.0, rho=0.1, pheromone_param=1.0, initial_pheromone=1.0, steps=100, nodes=None
        anthill = Anthill(mode=Anthill.ANT_MODE, alpha=config[0]['alpha'], beta=config[0]['beta'], colony_size=config[0]['colony_size'],
                          pheromone_param=config[0]['pheromone_param'], steps=config[0]['steps'], nodes=nodes, rho=config[0]['rho'])
        ant_distances.append(anthill.run())
        ant_times.append(time.time() - start_time_ant)
        anthill.plot(iteration=i)
        start_time_min_max = time.time()
        max_min_anthill = Anthill(mode=Anthill.MIN_MAX_MODE, colony_size=colony_size,
                                  steps=steps, nodes=nodes)
        min_max_distances.append(max_min_anthill.run())
        min_max_times.append(time.time() - start_time_min_max)
        max_min_anthill.plot(iteration=i)
    print('ant', ant_distances, ant_times)
    print('min max', min_max_distances, min_max_times)
    statistic_plot(ant_distances, min_max_distances, iteratios)
    statistic_plot(ant_times, min_max_times, iteratios)
