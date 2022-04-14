#!/usr/bin/env python
import csv
import math
import os
import warnings
import logging
from abc import ABCMeta, abstractmethod

import networkx as nx
import numpy as np

import itertools
import matplotlib.pyplot as plt
from scipy.stats import stats

from mapel.main.embedding.kamada_kawai.kamada_kawai import KamadaKawai

COLORS = []

from mapel.main.objects.Family import Family
import mapel.elections._print as pr

try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.manifold import Isomap
except ImportError as error:
    MDS = None
    TSNE = None
    SpectralEmbedding = None
    LocallyLinearEmbedding = None
    Isomap = None
    print(error)


class Experiment:
    __metaclass__ = ABCMeta
    """Abstract set of instances."""

    def __init__(self, instances=None, distances=None, dim=2, store=True,
                 coordinates=None, distance_id='emd-positionwise', experiment_id=None,
                 instance_type='ordinal', _import=True, clean=False, coordinates_names=None):

        self._import = _import
        self.clean = clean
        self.experiment_id = experiment_id

        if clean:
            self.clean_elections()

        self.distance_id = distance_id

        self.instances = None
        self.distances = None
        self.coordinates = None
        self.coordinates_lists = {}

        self.families = {}
        self.times = {}
        self.stds = {}
        self.matchings = {}
        self.features = {}
        self.coordinates_by_families = {}

        self.num_families = None
        self.num_instances = None
        self.main_order = None
        self.instance_type = instance_type

        if experiment_id is None:
            self.experiment_id = 'virtual'
            self.store = False
        else:
            self.store = True
            self.experiment_id = experiment_id
            self.create_structure()
            self.families = self.import_controllers()
            self.store = store

        if isinstance(instances, dict):
            self.instances = instances
            print('=== Omitting import! ===')
        elif _import and self.experiment_id != 'virtual':
            try:
                self.instances = self.add_instances_to_experiment()
                self.num_instances = len(self.instances)
                print('=== Elections imported successfully! ===')
            except FileNotFoundError:
                print('=== Elections not found! ===')
                self.instances = {}
        else:
            self.instances = {}

        if isinstance(distances, dict):
            self.distances = distances
            print('=== Omitting import! ===')
        elif _import and self.experiment_id != 'virtual':
            try:
                self.distances, self.times, self.stds = self.add_distances_to_experiment()
                print('=== Distances imported successfully! ===')
            except FileNotFoundError:
                print('=== Distances not found! ===')
        else:
            self.distances = {}

        if isinstance(coordinates, dict):
            self.coordinates = coordinates
            print('=== Omitting import! ===')
        elif _import and self.experiment_id != 'virtual':
            try:
                if coordinates_names is not None:
                    for file_name in coordinates_names:
                        self.coordinates_lists[file_name] = \
                            self.add_coordinates_to_experiment(dim=dim, file_name=file_name)
                    self.coordinates = self.coordinates_lists[coordinates_names[0]]
                else:
                    self.coordinates = self.add_coordinates_to_experiment(dim=dim)
                print('=== Coordinates imported successfully! ===')
            except FileNotFoundError:
                print('=== Coordinates not found! ===')
        else:
            self.coordinates = {}

        for family_id in self.families:
            for instance_id in self.families[family_id].instance_ids:
                self.instances[instance_id].label = self.families[family_id].label

    @abstractmethod
    def prepare_instances(self):
        pass

    @abstractmethod
    def add_instance(self):
        pass

    @abstractmethod
    def add_family(self):
        pass

    def embed(self, algorithm: str = 'spring', num_iterations: int = 1000, radius: float = np.infty,
              dim: int = 2, num_neighbors: int = None, method: str = 'standard',
              zero_distance: float = 0.1, factor: float = 1., saveas: str = None,
              init_pos: dict = None, fixed=True) -> None:

        if algorithm == 'spring':
            attraction_factor = 2
        else:
            attraction_factor = 1

        num_elections = len(self.distances)

        x = np.zeros((num_elections, num_elections))

        initial_positions = None

        if init_pos is not None:
            initial_positions = {}
            for i, instance_id_1 in enumerate(self.distances):
                if instance_id_1 in init_pos:
                    # initial_positions.update(0='portal')
                    initial_positions[i] = init_pos[instance_id_1]

        # print(initial_positions)

        for i, instance_id_1 in enumerate(self.distances):
            for j, instance_id_2 in enumerate(self.distances):
                if i < j:
                    self.distances[instance_id_1][instance_id_2] *= factor
                    if self.distances[instance_id_1][instance_id_2] == 0.:
                        self.distances[instance_id_1][instance_id_2] = zero_distance
                        self.distances[instance_id_2][instance_id_1] = zero_distance
                    if algorithm in {'spring'}:
                        normal = True
                        if self.distances[instance_id_1][instance_id_2] > radius:
                            x[i][j] = 0.
                            normal = False
                        if num_neighbors is not None:
                            tmp = self.distances[instance_id_1]
                            sorted_list_1 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            tmp = self.distances[instance_id_2]
                            sorted_list_2 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            if (instance_id_1 not in sorted_list_2[0:num_neighbors]) and (
                                    instance_id_2 not in sorted_list_1[0:num_neighbors]):
                                x[i][j] = 0.
                                normal = False
                        if normal:
                            x[i][j] = 1. / self.distances[instance_id_1][
                                instance_id_2]
                    else:
                        x[i][j] = self.distances[instance_id_1][instance_id_2]
                    x[i][j] = x[i][j] ** attraction_factor
                    x[j][i] = x[i][j]

        dt = [('weight', float)]
        y = x.view(dt)
        graph = nx.from_numpy_array(y)

        if num_neighbors is None:
            num_neighbors = 100

        if algorithm == 'spring':
            my_pos = nx.spring_layout(graph, iterations=num_iterations, dim=dim)
        elif algorithm in {'mds', 'MDS'}:
            my_pos = MDS(n_components=dim, dissimilarity='precomputed').fit_transform(x)
        elif algorithm in {'tsne', 'TSNE'}:
            my_pos = TSNE(n_components=dim).fit_transform(x)
        elif algorithm in {'se', 'SE'}:
            my_pos = SpectralEmbedding(n_components=dim).fit_transform(x)
        elif algorithm in {'isomap', 'ISOMAP'}:
            my_pos = Isomap(n_components=dim, n_neighbors=num_neighbors).fit_transform(x)
        elif algorithm in {'lle', 'LLE'}:
            my_pos = LocallyLinearEmbedding(n_components=dim,
                                            n_neighbors=num_neighbors,
                                            max_iter=num_iterations,
                                            method=method).fit_transform(x)
        elif algorithm.lower() in {'kamada-kawai', 'kamada'}:
            my_pos = KamadaKawai().embed(
                distances=x, initial_positions=initial_positions,
                fix_initial_positions=fixed
            )
        elif algorithm.lower() in {'geo'}:
            f1 = self.import_feature('voterlikeness_sqrt')
            f2 = self.import_feature('borda_diversity')
            for f in f1:
                if f1[f] is None:
                    f1[f] = 0
                if f2[f] is None:
                    f2[f] = 0
            my_pos = [[f1[e], f2[e]] for e in f1]
        else:
            my_pos = []
            logging.warning("Unknown method!")

        print(my_pos)

        coordinates = {}
        for i, instance_id in enumerate(self.distances):
            coordinates[instance_id] = [my_pos[i][d] for d in range(dim)]

        if self.store:
            if saveas is None:
                file_name = f'{self.distance_id}_{str(dim)}d.csv'
            else:
                file_name = saveas
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "coordinates", file_name)
            with open(path, 'w', newline='') as csvfile:

                writer = csv.writer(csvfile, delimiter=';')
                if dim == 2:
                    writer.writerow(["instance_id", "x", "y"])
                    print(["instance_id", "x", "y"])
                elif dim == 3:
                    writer.writerow(["instance_id", "x", "y", "z"])

                ctr = 0
                for instance_id in self.instances:
                    x = round(coordinates[instance_id][0], 5)
                    y = round(coordinates[instance_id][1], 5)
                    if dim == 2:
                        writer.writerow([instance_id, x, y])
                    elif dim == 3:
                        z = round(my_pos[ctr][2], 5)
                        writer.writerow([instance_id, x, y, z])
                    ctr += 1

        self.coordinates = coordinates

    def print_map(self, dim: int = 2, **kwargs) -> None:
        """ Print the two-dimensional embedding of multi-dimensional map of the instances """
        if dim == 2:
            pr.print_map_2d(self, **kwargs)
        elif dim == 3:
            pr.print_map_3d(self, **kwargs)

    def print_matrix(self, **kwargs):
        pr.print_matrix(experiment=self, **kwargs)

    @abstractmethod
    def add_instances_to_experiment(self):
        pass

    @abstractmethod
    def add_folders_to_experiment(self):
        pass

    @abstractmethod
    def create_structure(self):
        pass

    @abstractmethod
    def import_controllers(self):
        pass

    def add_coordinates_to_experiment(self, dim=2, file_name=None) -> dict:
        """ Import from a file precomputed coordinates of all the points --
        each point refer to one instance """

        coordinates = {}
        if file_name is None:
            file_name = f'{self.distance_id}_{dim}d.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", file_name)
        with open(path, 'r', newline='') as csv_file:

            # ORIGINAL
            reader = csv.DictReader(csv_file, delimiter=';')

            warn = False

            for row in reader:
                try:
                    instance_id = row['instance_id']
                except:
                    try:
                        instance_id = row['election_id']
                    except:
                        pass

                if dim == 2:
                    coordinates[instance_id] = [float(row['x']), float(row['y'])]
                elif dim == 3:
                    coordinates[instance_id] = [float(row['x']), float(row['y']), float(row['z'])]

                if instance_id not in self.instances:
                    warn = True

            # if warn:
            #     text = f'Possibly outdated coordinates are imported!'
            #     warnings.warn(text, stacklevel=2)

            if warn:
                text = f'Possibly outdated coordinates are imported!'
                logging.warning(text)
                # warnings.warn(text)

        return coordinates

    def compute_coordinates_by_families(self) -> None:
        """ Group all points by their families """

        coordinates_by_families = {}

        if self.families is None:
            self.families = {}
            for i, instance_id in enumerate(self.instances):
                ele = self.instances[instance_id]
                model = ele.model_id
                family_id = model
                label = instance_id
                color = COLORS[int(i % len(COLORS))]

                alpha = 1.

                self.families[instance_id] = Family(
                    model_id=model, family_id=family_id,
                    label=label, alpha=alpha,
                    color=color)

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(3)]
                coordinates_by_families[family_id][0].append(self.coordinates[family_id][0])
                coordinates_by_families[family_id][1].append(self.coordinates[family_id][1])
                try:
                    coordinates_by_families[family_id][2].append(self.coordinates[family_id][2])
                except Exception:
                    pass
        else:

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(3)]

                try:
                    for instance_id in self.families[family_id].instance_ids:
                        coordinates_by_families[family_id][0].append(self.coordinates[instance_id][0])
                        coordinates_by_families[family_id][1].append(self.coordinates[instance_id][1])
                        try:
                            coordinates_by_families[family_id][2].append(
                                self.coordinates[instance_id][2])
                        except Exception:
                            pass
                except:
                    for instance_id in self.families[family_id].instance_ids:
                        coordinates_by_families[family_id][0].append(
                            self.coordinates[instance_id][0])
                        coordinates_by_families[family_id][1].append(
                            self.coordinates[instance_id][1])
                        try:
                            coordinates_by_families[family_id][2].append(
                                self.coordinates[instance_id][2])
                        except Exception:
                            pass
        self.coordinates_by_families = coordinates_by_families

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.coordinates[i][d] - self.coordinates[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle) -> None:
        """ Rotate all the points by a given angle """

        for instance_id in self.instances:
            self.coordinates[instance_id][0], self.coordinates[instance_id][1] = \
                self.rotate_point(0.5, 0.5, angle, self.coordinates[instance_id][0],
                                  self.coordinates[instance_id][1])

        self.compute_coordinates_by_families()

    def reverse(self) -> None:
        """ Reverse all the points"""

        for instance_id in self.instances:
            self.coordinates[instance_id][0] = self.coordinates[instance_id][0]
            self.coordinates[instance_id][1] = -self.coordinates[instance_id][1]

        self.compute_coordinates_by_families()

    def update(self) -> None:
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_id}_2d.csv')

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(["instance_id", "x", "y"])

            for election_id in self.instances:
                x = round(self.coordinates[election_id][0], 5)
                y = round(self.coordinates[election_id][1], 5)
                writer.writerow([election_id, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py) -> (float, float):
        """ Rotate two-dimensional point by an angle """

        s, c = math.sin(angle), math.cos(angle)
        px -= cx
        py -= cy
        x_new, y_new = px * c - py * s, px * s + py * c
        px, py = x_new + cx, y_new + cy

        return px, py

    def add_distances_to_experiment(self) -> (dict, dict, dict):
        """ Import precomputed distances between each pair of instances from a file """
        file_name = f'{self.distance_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'distances', file_name)

        distances = {}
        times = {}
        stds = {}

        with open(path, 'r', newline='') as csv_file:

            reader = csv.DictReader(csv_file, delimiter=';')
            warn = False

            for row in reader:
                try:
                    instance_id_1 = row['election_id_1']
                    instance_id_2 = row['election_id_2']
                except:
                    try:
                        instance_id_1 = row['instance_id_1']
                        instance_id_2 = row['instance_id_2']
                    except:
                        pass

                if instance_id_1 not in self.instances or instance_id_2 not in self.instances:
                    continue

                if instance_id_1 not in distances:
                    distances[instance_id_1] = {}
                if instance_id_1 not in times:
                    times[instance_id_1] = {}
                if instance_id_1 not in stds:
                    stds[instance_id_1] = {}

                if instance_id_2 not in distances:
                    distances[instance_id_2] = {}
                if instance_id_2 not in times:
                    times[instance_id_2] = {}
                if instance_id_2 not in stds:
                    stds[instance_id_2] = {}

                try:
                    distances[instance_id_1][instance_id_2] = float(row['distance'])
                    distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
                except KeyError:
                    pass

                try:
                    times[instance_id_1][instance_id_2] = float(row['time'])
                    times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]
                except KeyError:
                    pass

                try:
                    stds[instance_id_1][instance_id_2] = float(row['std'])
                    stds[instance_id_2][instance_id_1] = stds[instance_id_1][instance_id_2]
                except KeyError:
                    pass

                if instance_id_1 not in self.instances:
                    warn = True

            if warn:
                text = f'Possibly outdated distances are imported!'
                warnings.warn(text)

        return distances, times, stds

    def clean_elections(self):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
        for file_name in os.listdir(path):
            os.remove(os.path.join(path, file_name))

    def get_feature(self, feature_id):

        # if feature_id not in self.features:
        #     self.features[feature_id] = self.import_feature(feature_id)

        self.features[feature_id] = self.import_feature(feature_id)

        return self.features[feature_id]

    def import_feature(self, feature_id):
        return pr.get_values_from_csv_file(self, feature_id=feature_id)

    def normalize_feature_by_feature(self, nom=None, denom=None, saveas=None):

        f1 = self.get_feature(nom)
        f2 = self.get_feature(denom)
        f3 = {}

        for election_id in f1:
            if f1[election_id] is None:
                f3[election_id] = None
            else:
                f3[election_id] = f1[election_id] / f2[election_id]

        self.store_feature(feature_dict=f3, saveas=saveas)

    def store_feature(self, feature_dict=None, saveas=None):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "features", f'{saveas}.csv')

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["election_id", "value"])
            # writer.writerow(["election_id", "value", "bound", "num_large_parties"])
            for key in feature_dict:
                writer.writerow([key, str(feature_dict[key])])

    def import_distances(self, distance_id):

        distances = {}

        file_name = f'{distance_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'distances', file_name)

        with open(path, 'r', newline='') as csv_file:

            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                try:
                    instance_id_1 = row['election_id_1']
                    instance_id_2 = row['election_id_2']
                except:
                    try:
                        instance_id_1 = row['instance_id_1']
                        instance_id_2 = row['instance_id_2']
                    except:
                        pass

                if instance_id_1 not in self.instances or instance_id_2 not in self.instances:
                    continue

                if instance_id_1 not in distances:
                    distances[instance_id_1] = {}

                if instance_id_2 not in distances:
                    distances[instance_id_2] = {}

                try:
                    distances[instance_id_1][instance_id_2] = float(row['distance'])
                    distances[instance_id_2][instance_id_1] = distances[instance_id_1][
                        instance_id_2]
                except KeyError:
                    pass

        return distances

    def print_correlation(self, distance_id_1='spearman', distance_id_2='l1-mutual_attraction',
                          title=None):

        all_distances = {}

        all_distances[distance_id_1] = self.import_distances(distance_id=distance_id_1)
        all_distances[distance_id_2] = self.import_distances(distance_id=distance_id_2)

        names = list(all_distances.keys())

        def nice(name):
            return{
                'spearman': 'Spearman',
                'l1-mutual_attraction': '$\ell_1$ Mutual Attraction',
                'emd-positionwise': 'EMD-Positionwise',
                }.get(name)

        def normalize(name):
            return{
                'spearman': 1./12,
                'l1-mutual_attraction': 1./12,
                'emd-positionwise': 1.,
                }.get(name)


        for name_1, name_2 in itertools.combinations(names, 2):

            # for target in ['double', 'radius', '2g-IC', 'IC_', '1d_', 'maleuc', 'malasym', 'vec']:
            #         ['MD', 'MA', 'CH', 'ID', '2d_rev', 'Mallows']:
            # for target in ['MD']:

            values_x = []
            values_y = []
            empty_x = []
            empty_y = []
            for e1, e2 in itertools.combinations(all_distances[name_1], 2):
                # if e1 in ['IC'] or e2 in ['IC']:
                # if target in e1 or target in e2:
                #     empty_x.append(all_distances[name_1][e1][e2])
                #     empty_y.append(all_distances[name_2][e1][e2])
                # else:
                values_x.append(all_distances[name_1][e1][e2] * normalize(name_1))
                values_y.append(all_distances[name_2][e1][e2] * normalize(name_2))

            fig = plt.figure()
            ax = fig.add_subplot()

            ax.scatter(values_x, values_y, s=4, alpha=0.01, color='purple')
            # ax.scatter(empty_x, empty_y, s=8, alpha=0.2, color='blue')

            pear = round(stats.pearsonr(values_x, values_y)[0], 3)
            pear_text = f'PCC = {pear}'
            print('pear', pear)
            # plt.text(0.7, 0.1, pear_text, transform=ax.transAxes, size=14)

            if title is None:
                title = f'{nice(name_1)} vs {nice(name_2)}'
            # title=f'{target}'

            plt.xlabel(nice(name_1), size=20)
            plt.ylabel(nice(name_2), size=20)
            # plt.title(title, size=24)
            # saveas = f'images/correlation/corr_{name_1}_{name_2}_{target}'
            saveas = f'images/correlation/corr_{name_1}_{name_2}'
            plt.savefig(saveas, bbox_inches='tight')
            plt.show()
