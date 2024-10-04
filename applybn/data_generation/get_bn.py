from bamt.preprocessors import Preprocessor
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.composite_bn import CompositeBN
from copy import copy
from bamt.utils.composite_utils.CompositeModel import CompositeModel, CompositeNode


def get_hybrid_bn(data, n_bins=10, mixtures=True):
    cur_data = copy(data)

    encoder = LabelEncoder()
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    pp = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
    discrete_data, est = pp.apply(cur_data)

    nodes_data = pp.info

    bn = HybridBN(use_mixture=mixtures, has_logit=False)
    bn.add_nodes(nodes_data)
    bn.add_edges(data=discrete_data,
                 scoring_function=("K2", ),
                 )

    bn.fit_parameters(cur_data)

    return bn


def get_parents(node, structure):
    parents = []
    for edge in structure:
        if (edge[1] == node) & (edge[0] not in parents):
            parents.append(edge[0])
    return parents


def get_children(node, structure):
    children = []
    for edge in structure:
        if (edge[0] == node) & (edge[1] not in children):
            children.append(edge[1])
    return children


def get_compose_bn(data, n_bins=10, mixtures=True):
    cur_data = copy(data)

    encoder = LabelEncoder()
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    p = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
    discrete_data, est = p.apply(cur_data)

    nodes_data = p.info

    bn = HybridBN(use_mixture=mixtures, has_logit=False)
    bn.add_nodes(nodes_data)
    bn.add_edges(data=discrete_data,
                 scoring_function=("K2",),
                 )

    composite_nodes = []
    for c in data.columns:
        node = CompositeNode(
            nodes_from=None,
            content={
                "name": c,
                "type": p.nodes_types[c],
                "parent_model": None,
            },
        )
        composite_nodes.append(node)

    for node in composite_nodes:
        node_parents = get_parents(node.content['name'], bn.edges)
        if node_parents:
            node.nodes_from = []
            for parent in node_parents:
                for node_p in composite_nodes:
                    if node_p.content["name"] == parent:
                        node.nodes_from.append(node_p)
                        break
            if node.content['type'] == 'cont':
                node.content["parent_model"] = "LinearRegression"
            else:
                node.content["parent_model"] = "LogisticRegression"

    cbn = CompositeBN()
    cbn.add_nodes(p.info)
    init_cbn = CompositeModel(nodes=composite_nodes)
    cbn.add_edges(data, custom_initial_structure=[init_cbn], verbose=False)
    cbn.fit_parameters(data)

    return bn
