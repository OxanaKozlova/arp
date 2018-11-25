import math
import random
from matplotlib import pyplot as plt
import time
from scipy.spatial.distance import cdist


class Hive:
    class Site:
        class Edge:
            def __init__(self, node_1, node_2, weight):
                self.node_1 = node_1
                self.node_2 = node_2
                self.weight = weight

        def __init__(self, nodes):
            self.num_nodes = len(nodes)
            self.nodes = nodes
            self.edges = [
                [None] * self.num_nodes for _ in range(self.num_nodes)]
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                        pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)))

        def get_tour(self):
            node_indexes = list(range(self.num_nodes))
            random.shuffle(node_indexes)
            return node_indexes

        def get_distance(self, tour):
            distance = 0.0
            for i in range(self.num_nodes):
                distance += self.edges[tour[i]
                                       ][tour[(i + 1) % self.num_nodes]].weight
            return distance

    class Bee:

        def __init__(self, site, neigh_dist=1):
            self.site = site
            self.tour = self.site.get_tour()
            self.distance = self.site.get_distance(self.tour)
            self.neigh_dist = neigh_dist

        def find_neigh_tour(self, selected_tour):
            neigh_tour = list(selected_tour)
            indexes = random.sample(
                range(0, self.site.num_nodes - 1), self.neigh_dist)
            for i in range(len(indexes) - 1):
                neigh_tour[indexes[i]], neigh_tour[indexes[i + 1]
                                                   ] = neigh_tour[indexes[i + 1]], neigh_tour[indexes[i]]
            return neigh_tour

        def change_tour(self, new_tour):
            new_tour_distance = self.site.get_distance(new_tour)
            if new_tour_distance < self.distance:
                self.tour = new_tour
                self.distance = new_tour_distance

        def find_tour(self):
            self.tour = self.site.get_tour()
            self.distance = self.site.get_distance(self.tour)

    def __init__(self, colony_size=10, num_scout_bees=10, num_best_bees=5, num_selected_bees=3, steps=100, nodes=None, neigh_dist=1, num_selected_tours=10, num_best_tours=2):
        self.steps = steps
        self.labels = range(1, len(nodes) + 1)
        self.site = self.Site(nodes)
        self.num_selected_tours = num_selected_tours
        self.num_best_tours = num_best_tours

        self.scout_bees = [
            self.Bee(site=self.site, neigh_dist=neigh_dist) for _ in range(num_scout_bees)]
        self.best_bees = [self.Bee(site=self.site, neigh_dist=neigh_dist)
                          for _ in range(num_best_bees)]
        self.selected_bees = [self.Bee(
            site=self.site, neigh_dist=neigh_dist) for _ in range(num_selected_bees)]

        self.global_best_tour = self.site.get_tour()
        self.global_best_distance = self.site.get_distance(
            self.global_best_tour)

    def investigate_tours(self, bees, tours):
        for selected_tour in tours:
            for bee in bees:
                neigh_tour = bee.find_neigh_tour(selected_tour)
                bee.change_tour(neigh_tour)

    def find_best_tour(self):
        employeed_bees = self.best_bees + self.selected_bees
        best_bee = min(employeed_bees, key=lambda bee: int(bee.distance))
        if best_bee.distance < self.global_best_distance:
            self.global_best_tour = best_bee.tour
            self.global_best_distance = best_bee.distance

    def update_population(self, indexes):
        all_indexes = list(range(len(self.scout_bees)))
        should_update_indexes = list(set(all_indexes) - set(indexes))
        for i in should_update_indexes:
            self.scout_bees[i].find_tour()

    def run(self):
        for step in range(self.steps):
            self.scout_bees = sorted(
                self.scout_bees, key=lambda scout_bee: scout_bee.distance)
            selected_tour_indexes = list(
                range(self.num_best_tours, self.num_selected_tours))
            selected_tours = [
                self.scout_bees[i].tour for i in selected_tour_indexes]
            self.investigate_tours(self.selected_bees, selected_tours)

            best_tours = [self.scout_bees[i].tour for i in range(
                self.num_best_tours)]
            self.investigate_tours(self.best_bees, best_tours)

            self.find_best_tour()
            num_best_tour_indexes = list(range(self.num_best_tours))
            self.update_population(
                num_best_tour_indexes + selected_tour_indexes)

        return self.global_best_distance

    def plot(self, annotation_size=8, dpi=120, iteration=0):
        x = [self.site.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.site.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=1)
        plt.scatter(x, y, s=math.pi * 4)
        plt.title('Bee tour')
        for i in self.global_best_tour:
            plt.annotate(self.labels[i],
                         self.site.nodes[i], size=annotation_size)

        name = 'plots/bee_tour_{0}.png'.format(iteration)
        plt.savefig(name, dpi=dpi)
        plt.gcf().clear()


def statistic_plot(distances, iterations, mode):
    plt.plot(range(1, iterations + 1), distances)
    plt.legend([mode], loc='upper left')
    name = 'plots/statistic.png'
    plt.savefig(name, dpi=120)
    plt.show()
    plt.gcf().clear()


if __name__ == '__main__':
    num_nodes = 25
    distances = []
    times = []
    nodes = [(random.uniform(-1500, 1500), random.uniform(-1500, 1500))
             for _ in range(0, num_nodes)]

# num_selected_bees < num_best_bees < num_scout_bees
# num_best_tours < num_selected_tours
    config = [
        {'steps': 200, 'num_scout_bees': 300, 'num_best_bees': 200, 'num_selected_bees': 80,
            'neigh_dist': 10, 'num_selected_tours': 80, 'num_best_tours': 50},
        {'steps': 300, 'num_scout_bees': 300, 'num_best_bees': 200, 'num_selected_bees': 80,
            'neigh_dist': 10, 'num_selected_tours': 80, 'num_best_tours': 50},
        {'steps': 200, 'num_scout_bees': 200, 'num_best_bees': 100, 'num_selected_bees': 80,
            'neigh_dist': 10, 'num_selected_tours': 80, 'num_best_tours': 40},
        {'steps': 200, 'num_scout_bees': 400, 'num_best_bees': 200, 'num_selected_bees': 80,
            'neigh_dist': 10, 'num_selected_tours': 80, 'num_best_tours': 50},
        {'steps': 200, 'num_scout_bees': 400, 'num_best_bees': 300, 'num_selected_bees': 100,
            'neigh_dist': 10, 'num_selected_tours': 80, 'num_best_tours': 50},
        {'steps': 200, 'num_scout_bees': 300, 'num_best_bees': 200, 'num_selected_bees': 80,
            'neigh_dist': 5, 'num_selected_tours': 100, 'num_best_tours': 60},
        {'steps': 300, 'num_scout_bees': 100, 'num_best_bees': 40, 'num_selected_bees': 20,
            'neigh_dist': 10, 'num_selected_tours': 20, 'num_best_tours': 15},
        {'steps': 200, 'num_scout_bees': 300, 'num_best_bees': 250, 'num_selected_bees': 150,
            'neigh_dist': 3, 'num_selected_tours': 100, 'num_best_tours': 80},
        {'steps': 100, 'num_scout_bees': 100, 'num_best_bees': 60, 'num_selected_bees': 40,
            'neigh_dist': 10, 'num_selected_tours': 60, 'num_best_tours': 30},
        {'steps': 300, 'num_scout_bees': 400, 'num_best_bees': 300, 'num_selected_bees': 200,
            'neigh_dist': 7, 'num_selected_tours': 100, 'num_best_tours': 80},
    ]

    for i in range(len(config)):
        hive = Hive(num_scout_bees=config[i]['num_scout_bees'], num_best_bees=config[i]['num_best_bees'],
                    num_selected_bees=config[i]['num_selected_bees'], steps=config[i]['steps'], nodes=nodes,
                    neigh_dist=config[i]['neigh_dist'], num_selected_tours=config[i]['num_selected_tours'],
                    num_best_tours=config[i]['num_best_tours'])
        start = time.time()
        distances.append(hive.run())
        times.append(time.time() - start)
        hive.plot(iteration=i)
    statistic_plot(distances, len(config), 'distances')
    statistic_plot(times, len(config), 'time')
