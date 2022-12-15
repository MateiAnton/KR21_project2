from BayesNet import *
from BNReasoner import *
import numpy as np

from network_builder import build_network

IJXYO = "testing/lecture_example2.BIFXML"


def approximately_equal(a, b, epsilon=0.0001):
    return abs(a - b) < epsilon


def test_prune_network():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    reasoner.prune_network(["I", "X"], {"J": True})
    vars = bn.get_all_variables()
    assert (
        "I" in vars
        and "X" in vars
        and "J" in vars
        and "Y" not in vars
        and "O" not in vars
    )
    assert "X" in bn.get_children("I")
    assert len(bn.get_parents("J")) == 0
    assert len(bn.get_parents("X")) == 1
    assert len(bn.get_parents("I")) == 0
    # assert that only I and p are in the CPT of I
    assert len(bn.get_cpt("I").columns) == 2
    # assert that only X, I, and p are in the CPT of X
    assert len(bn.get_cpt("X").columns) == 3 and "I" in bn.get_cpt("X").columns
    # assert that only J and p are in the CPT of J
    assert len(bn.get_cpt("J").columns)


def test_multiply_factors():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    factor1 = bn.get_cpt("I")
    factor2 = bn.get_cpt("X")
    multiplied = reasoner.multiply_factors(factor1, factor2)
    # for shared variables
    # assert that the only columns are I, J, X, and p
    assert (
        len(multiplied.columns) == 4
        and "J" in multiplied.columns
        and "I" in multiplied.columns
        and "X" in multiplied.columns
    )
    factor2 = bn.get_cpt("J")
    multiplied = reasoner.multiply_factors(factor1, factor2)
    # for distinct variables
    # assert that the only columns are I, J, and p
    assert (
        len(multiplied.columns) == 3
        and "J" in multiplied.columns
        and "I" in multiplied.columns
    )


def test_min_degree_ordering():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    ordering = reasoner.min_degree_ordering(bn.get_all_variables())
    interaction_graph = bn.get_interaction_graph()
    while len(ordering) > 0:
        var = ordering.pop(0)
        # check that var has the minimum amount of children in interaction graph from all nodes in ordering
        for other_var in ordering:
            assert interaction_graph.degree(var) <= interaction_graph.degree(other_var)
        # connect the neighbours of min_degree_node
        neighbours = list(interaction_graph.neighbors(var))
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                interaction_graph.add_edge(neighbours[i], neighbours[j])
        interaction_graph.remove_node(var)


def test_min_fill_ordering():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    ordering = reasoner.min_fill_ordering(bn.get_all_variables())
    interaction_graph = bn.get_interaction_graph()
    while len(ordering) > 0:
        var = ordering.pop(0)
        # check that var has the minimum amount of children in interaction graph from all nodes in ordering
        for other_var in ordering:
            assert reasoner._nr_of_edges_that_a_variable_would_remove(
                var, interaction_graph
            ) <= reasoner._nr_of_edges_that_a_variable_would_remove(
                other_var, interaction_graph
            )
        # connect the neighbours of min_degree_node
        neighbours = list(interaction_graph.neighbors(var))
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                interaction_graph.add_edge(neighbours[i], neighbours[j])
        interaction_graph.remove_node(var)


def test_sum_out():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    factor = bn.get_cpt("X")
    summed_out = reasoner.sum_out(["J", "I"], factor)
    assert len(summed_out.columns) == 2 and "X" in summed_out.columns
    # assert correct value for P(X=True)
    assert approximately_equal(summed_out.loc[summed_out["X"] == True]["p"].item(), 1.1)
    # assert correct value for P(X=False)
    assert approximately_equal(
        summed_out.loc[summed_out["X"] == False]["p"].item(), 2.9
    )


def test_max_out():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    factor = bn.get_cpt("X")
    maxed_out = reasoner.max_out(["J", "I", "X"], factor)
    # assert correct value
    assert approximately_equal(maxed_out["p"].item(), 0.95)


def test_variable_elimination():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    factors = reasoner.variable_elimination(["I", "J", "Y", "O", "X"])
    assert len(factors) == 1
    vals = factors[0]["p"].values.tolist()
    assert approximately_equal(vals[0], 0.725)
    assert approximately_equal(vals[1], 0.275)


def test_compute_marginal_distribution():
    pass
    # TODO
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    end_factor = reasoner.compute_marginal_distribution(["O", "X"], {"J": True})
    assert len(end_factor.columns) == 3
    # assert probabilities sum to 1
    assert approximately_equal(end_factor["p"].sum(), 1)
    # assert correct values for P(O=True, X=True)
    assert approximately_equal(
        end_factor.loc[(end_factor["O"] == True) & (end_factor["X"] == True)][
            "p"
        ].item(),
        0.49,
    )
    # assert correct values for P(O=True, X=False)
    assert approximately_equal(
        end_factor.loc[(end_factor["O"] == True) & (end_factor["X"] == False)][
            "p"
        ].item(),
        0.0148,
    )
    # assert correct values for P(O=False, X=True)
    assert approximately_equal(
        end_factor.loc[(end_factor["O"] == False) & (end_factor["X"] == True)][
            "p"
        ].item(),
        0.01,
    )
    # assert correct values for P(O=False, X=False)
    assert approximately_equal(
        end_factor.loc[(end_factor["O"] == False) & (end_factor["X"] == False)][
            "p"
        ].item(),
        0.4852,
    )


def test_maximum_a_posteriori():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    prob, values = reasoner.maximum_a_posteriori(["I", "J"], {"O": True})
    assert approximately_equal(prob, 0.242720)
    assert values["J"] == False


def test_most_probable_explanation():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    prob, values = reasoner.most_probable_explanation({"J": True, "O": False})
    assert approximately_equal(prob, 0.230422)
    assert values["J"] == True
    assert values["I"] == False
    assert values["X"] == False
    assert values["Y"] == False
    assert values["O"] == False


test_prune_network()
test_multiply_factors()
test_min_degree_ordering()
test_min_fill_ordering()
test_sum_out()
test_max_out()
test_variable_elimination()
test_compute_marginal_distribution()
test_maximum_a_posteriori()
test_most_probable_explanation()
