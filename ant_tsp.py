import math
import random
from matplotlib import pyplot as plt
import numpy as np
import time


class Anthill:
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

    def __init__(self, colony_size=10, min_param=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_param=1.0, initial_pheromone=1.0, steps=100, nodes=None):
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
        return self.global_best_distance

    def plot(self, annotation_size=8, dpi=120, iteration=0):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=1)
        plt.scatter(x, y, s=math.pi * 4)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)

        name = 'plots/tour_{0}.png'.format(iteration)
        plt.savefig(name, dpi=dpi)
        plt.gcf().clear()


def statistic_plot(ant_distances, iterations, mode):
    plt.plot(list(range(1, iterations + 1)), ant_distances)

    plt.legend([mode], loc='upper left')
    name = 'plots/statistic.png'
    plt.savefig(name, dpi=120)
    plt.show()
    plt.gcf().clear()


if __name__ == '__main__':
    colony_size = 50
    steps = 50
    nodes_count = 100

    ant_distances = []
    ant_times = []

    iteratios = 20
    config = [
        {'steps': 50, 'colony_size': 50, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'colony_size': 50, 'alpha': 2.0,
            'beta': 0.5, 'pheromone_param': 1.0, 'rho': 0.2},
        {'steps': 200, 'colony_size': 100, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.2},
        {'steps': 100, 'colony_size': 100, 'alpha': 1.0,
            'beta': 7.0, 'pheromone_param': 1.0, 'rho': 0.3},
        {'steps': 100, 'colony_size': 100, 'alpha': 0.5,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.5},
        {'steps': 50, 'colony_size': 100, 'alpha': 2.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'colony_size': 100, 'alpha': 1.0,
            'beta': 0.5, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 80, 'colony_size': 50, 'alpha': 1.0,
            'beta': 2.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 50, 'colony_size': 100, 'alpha': 8.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 100, 'colony_size': 20, 'alpha': 1.0,
            'beta': 5.0, 'pheromone_param': 1.0, 'rho': 0.9},
        {'steps': 10, 'colony_size': 200, 'alpha': 2.0,
            'beta': 3.0, 'pheromone_param': 0.5, 'rho': 0.1},
        {'steps': 100, 'colony_size': 100, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 2.0, 'rho': 0.5},
        {'steps': 100, 'colony_size': 150, 'alpha': 0.1,
            'beta': 0.1, 'pheromone_param': 1.0, 'rho': 0.5},
        {'steps': 100, 'colony_size': 100, 'alpha': 2.0,
            'beta': 2.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 20, 'colony_size': 20, 'alpha': 1.0,
            'beta': 1.0, 'pheromone_param': 5.0, 'rho': 0.5},
        {'steps': 100, 'colony_size': 200, 'alpha': 2.0,
            'beta': 1.0, 'pheromone_param': 1.0, 'rho': 0.1},
        {'steps': 200, 'colony_size': 100, 'alpha': 0.0,
            'beta': 1.0, 'pheromone_param': 2.0, 'rho': 0.7},
        {'steps': 100, 'colony_size': 100, 'alpha': 1.0,
            'beta': 0.0, 'pheromone_param': 0.5, 'rho': 0.1},
        {'steps': 100, 'colony_size': 100, 'alpha': 1.0,
            'beta': 2.0, 'pheromone_param': 2.0, 'rho': 0.1},
        {'steps': 100, 'colony_size': 100, 'alpha': 0.1,
            'beta': 0.1, 'pheromone_param': 2.0, 'rho': 0.1},
    ]
    nodes = [(random.uniform(-1500, 1500), random.uniform(-1500, 1500))
             for _ in range(0, nodes_count)]
    for i in range(iteratios):
        start_time_ant = time.time()
        # colony_size=10, min_param=0.001, alpha=1.0, beta=3.0, rho=0.1, pheromone_param=1.0, initial_pheromone=1.0, steps=100, nodes=None
        anthill = Anthill(alpha=config[0]['alpha'], beta=config[0]['beta'], colony_size=config[0]['colony_size'],
                          pheromone_param=config[0]['pheromone_param'], steps=config[0]['steps'], nodes=nodes, rho=config[0]['rho'])
        ant_distances.append(anthill.ant_process())
        ant_times.append(time.time() - start_time_ant)
        anthill.plot(iteration=i)
    print('ant', ant_distances, ant_times)
    statistic_plot(ant_distances, iteratios, 'distances')
    statistic_plot(ant_times, iteratios, 'time')
