import random
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score, BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination


class BNFeatureGenerator:
    def __init__(self, known_structure=None):
        self.known_structure = known_structure
        self.bn = None
        self.variables = None
        self.num_classes = None

    def fit(self, data, target=None, black_list=None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        self.variables = data.columns.tolist()

        if target is not None:
            if not isinstance(target, pd.Series):
                target = pd.Series(target)
            self.num_classes = len(target.unique())

        if self.known_structure:
            self.bn = BayesianNetwork(self.known_structure)
            self.bn.fit(data, estimator=BayesianEstimator)
        else:
            constructor = self._BayesianNetworkConstructor(data, black_list)
            self.bn = constructor.construct_network()

    def generate_features(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        inference = VariableElimination(self.bn)
        new_features = []

        for _, instance in data.iterrows():
            instance_features = []
            for feature in self.variables:
                evidence = {f: instance[f] for f in self.bn.get_parents(feature)
                            if f in instance.index}
                try:
                    prob = inference.query([feature], evidence=evidence)
                    lambda_value = prob.values[instance[feature]]
                except:
                    lambda_value = 1 / self.num_classes if self.num_classes else 0.5
                instance_features.append(lambda_value)
            new_features.append(instance_features)

        return pd.DataFrame(new_features, columns=[f'lambda_{c}' for c in self.variables])

    class _BayesianNetworkConstructor:
        def __init__(self, data, black_list=None):
            self.data = data
            self.variables = data.columns.tolist()
            self.repository = []
            self.max_repository_size = 10
            self.initial_max_arcs = len(self.variables) // 2
            self.black_list = black_list or []

        def is_valid_edge(self, edge):
            return edge not in self.black_list

        def create_forest_structure(self):
            nodes = self.variables.copy()
            edges = []
            used_nodes = {nodes.pop(0)}

            while nodes:
                child = nodes.pop(0)
                valid_parents = [p for p in used_nodes
                                 if self.is_valid_edge((p, child))]
                if valid_parents:
                    parent = random.choice(valid_parents)
                    edges.append((parent, child))
                    used_nodes.add(child)
                else:
                    used_nodes.add(child)

            return BayesianNetwork(edges)

        def calculate_network_score(self, network, sample_data):
            score = BicScore(data=sample_data)
            total_score = 0
            for node in network.nodes():
                parents = list(network.get_parents(node))
                penalty = len(parents) * 0.1
                total_score += score.local_score(node, parents) - penalty
            return total_score

        def modify_network(self, network, max_arcs):
            edges = list(network.edges())
            if not edges:
                return network

            operation = random.choice(['add', 'delete', 'reverse'])
            new_edges = edges.copy()

            if operation == 'add' and len(edges) < max_arcs:
                attempts = 0
                while attempts < 10:
                    node1, node2 = random.sample(self.variables, 2)
                    if ((node1, node2) not in edges and
                            (node2, node1) not in edges and
                            self.is_valid_edge((node1, node2))):
                        new_edges.append((node1, node2))
                        break
                    attempts += 1

            elif operation == 'delete' and edges:
                edge = random.choice(edges)
                new_edges.remove(edge)

            elif operation == 'reverse' and edges:
                edge = random.choice(edges)
                if self.is_valid_edge((edge[1], edge[0])):
                    new_edges.remove(edge)
                    new_edges.append((edge[1], edge[0]))

            try:
                new_network = BayesianNetwork(new_edges)
                new_network.check_model()
                return new_network
            except:
                return network

        def construct_network(self, iterations=100):
            current_network = self.create_forest_structure()
            sample_size = max(len(self.data) // 10, 2)
            max_arcs = self.initial_max_arcs
            current_score = self.calculate_network_score(current_network,
                                                         self.data.sample(n=sample_size))

            for i in range(iterations):
                if i % (iterations // 4) == 0:
                    sample_size = min(sample_size * 2, len(self.data))
                    max_arcs = min(max_arcs + 2, len(self.variables) * 2)

                sample_data = self.data.sample(n=sample_size)
                new_network = self.modify_network(current_network, max_arcs)
                new_score = self.calculate_network_score(new_network, sample_data)

                if new_score > current_score:
                    current_network = new_network
                    current_score = new_score
                    self.repository.append((new_network, new_score))
                    self.repository.sort(key=lambda x: x[1], reverse=True)
                    self.repository = self.repository[:self.max_repository_size]
                elif self.repository:
                    current_network, current_score = random.choice(self.repository)
                    current_network = self.modify_network(current_network, max_arcs)

            current_network.fit(self.data, estimator=BayesianEstimator)
            return current_network
