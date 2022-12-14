from ast import Tuple
from typing import Union
from BayesNet import BayesNet
from typing import List, Dict
import pandas as pd
import networkx as nx
import itertools
from collections import deque


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def prune_network(self, vars: List[str], e: Dict[str, bool]):
        if not all(var in self.bn.get_all_variables() for var in vars):
            raise Exception("provided variables not in network")
        if not all(key in self.bn.get_all_variables() for key in e.keys()):
            raise Exception("provided evidence not part of netowrk")

        for var in e.keys():
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))
                cpt = self.bn.get_cpt(child)
                e_without_factor = e.copy()
                if child in e:
                    del e_without_factor[child]
                cpt = self.bn.reduce_factor_rem_row(pd.Series(e_without_factor), cpt)
                cpt = cpt.drop(columns=[var])
                self.bn.update_cpt(child, cpt)

        self._prune_leafs_rec(list(set(vars).union(set(e.keys()))))

    def _prune_leafs_rec(self, target_vars: List[str]):
        leafs = [
            node
            for node in self.bn.structure.nodes()
            if self.bn.structure.out_degree(node) == 0
        ]
        leafs_to_del = [leaf for leaf in leafs if leaf not in target_vars]
        for leaf in leafs_to_del:
            self.bn.del_var(leaf)
        if len(leafs_to_del) != 0:
            self._prune_leafs_rec(target_vars)

    def d_separated(self, X, Y, Z):
        """
        Checks if X and Y are d-separated given Z.
        """
        graph = self.bn.structure.copy()

        while True:
            leaves = [
                node
                for node in graph.nodes()
                if graph.out_degree(node) == 0
                and node not in Z
                and node not in X
                and node not in Y
            ]
            if len(leaves) == 0:
                break
            for leaf in leaves:
                graph.remove_node(leaf)

        for node in Z:
            for child in self.bn.get_children(node):
                graph.remove_edge(node, child)

        subgraphs = nx.weakly_connected_components(graph)
        for subgraph in subgraphs:
            if (
                len(set(X).intersection(subgraph)) > 0
                and len(set(Y).intersection(subgraph)) > 0
            ):
                return False
        return True

    def multiply_factors(
        self, factor1: pd.DataFrame, factor2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Multiplies two factors and returns the resulting factor.
        """

        if len(set(factor1.columns).intersection(set(factor2.columns))) > 1:
            merged = pd.merge(
                factor1.drop(columns=["p"]), factor2.drop(columns=["p"]), how="outer"
            )
        else:
            merged = pd.merge(
                factor1.drop(columns=["p"]), factor2.drop(columns=["p"]), how="cross"
            )
        # delete all rows with NaN
        merged = merged.dropna()

        columns1 = factor1.columns.values[:-1]
        columns2 = factor2.columns.values[:-1]
        ps = []
        for _, row in merged.iterrows():
            current_col_values1 = row[columns1]
            current_col_values2 = row[columns2]
            p1 = list(
                factor1.loc[
                    (
                        factor1[list(current_col_values1.to_dict())]
                        == current_col_values1
                    ).all(axis=1)
                ]["p"]
            )[0]
            p2 = list(
                factor2.loc[
                    (
                        factor2[list(current_col_values2.to_dict())]
                        == current_col_values2
                    ).all(axis=1)
                ]["p"]
            )[0]
            ps.append(p1 * p2)

        merged["p"] = ps

        return merged

    def min_degree_ordering(self, variables: List[str]) -> List[str]:
        interaction_graph = self.bn.get_interaction_graph()
        ordering = []
        while len(variables) > 0:
            # nx.draw(interaction_graph, with_labels=True, node_size=3000)
            # plt.show()

            # get min degree node from variables in interaction_graph
            min_degree = float("inf")
            for var in variables:
                if interaction_graph.degree(var) < min_degree:
                    min_degree = interaction_graph.degree(var)
                    min_degree_node = var

            variables.remove(min_degree_node)
            ordering.append(min_degree_node)
            # get neighbours of min_degree_node
            neighbours = list(interaction_graph.neighbors(min_degree_node))
            # connect the neighbours of min_degree_node
            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    interaction_graph.add_edge(neighbours[i], neighbours[j])
            interaction_graph.remove_node(min_degree_node)
        return ordering

    def min_fill_ordering(self, variables: List[str]) -> List[str]:
        interaction_graph = self.bn.get_interaction_graph()
        ordering = []
        while len(variables) > 0:
            # nx.draw(interaction_graph, with_labels=True, node_size=3000)
            # plt.show()

            min_fill = float("inf")
            for var in variables:
                neighbours = list((interaction_graph.neighbors(var)))
                nr_of_interactions = 0
                for i in range(len(neighbours)):
                    for j in range(i + 1, len(neighbours)):
                        if not interaction_graph.has_edge(neighbours[i], neighbours[j]):
                            nr_of_interactions += 1
                if nr_of_interactions < min_fill:
                    min_fill = nr_of_interactions
                    min_fill_node = var

            variables.remove(min_fill_node)
            ordering.append(min_fill_node)
            # get neighbours of min_degree_node
            neighbours = list(interaction_graph.neighbors(min_fill_node))
            # connect the neighbours of min_degree_node
            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    interaction_graph.add_edge(neighbours[i], neighbours[j])
            interaction_graph.remove_node(min_fill_node)
        return ordering

    def sum_out(self, vars: List[str], factor: pd.DataFrame) -> pd.DataFrame:
        """
        Sums out a variable from a factor and returns the resulting factor.
        """
        df = factor.drop(columns=vars)
        if len(df.columns) > 1:
            return df.groupby(df.columns.drop("p").tolist(), as_index=False).sum()
        else:
            return df

    def max_out(self, vars: List[str], factor: pd.DataFrame) -> pd.DataFrame:
        """
        Maxes out variables from a factor and returns the resulting extended factor.
        """
        sorted_df = factor.sort_values(by=["p"], ascending=False)
        if len(factor.columns.drop(["p"] + vars)) > 1:
            sorted_df = sorted_df.drop_duplicates(factor.columns.drop(["p"] + vars))
        else:
            sorted_df = sorted_df.iloc[:1]
        return sorted_df

    def variable_elimination(
        self, ordering: List[str], max_out=False
    ) -> List[pd.DataFrame]:
        """
        Sums out the variable in the given ordering and returns the resulting factor.
        """
        factors = self.bn.get_all_cpts()
        for var in ordering:
            factors_containing_var = [f for f in factors.items() if var in f[1].columns]
            # remove factors containing var
            for f in factors_containing_var:
                factors.pop(f[0])

            factors_containing_var = [f[1] for f in factors_containing_var]

            # multiply factors containing var
            while len(factors_containing_var) > 1:
                first_factor = factors_containing_var.pop()
                second_factor = factors_containing_var.pop()
                factors_containing_var.append(
                    self.multiply_factors(first_factor, second_factor),
                )
            if max_out:
                factors[var] = self.max_out([var], factors_containing_var[0])
            else:
                factors[var] = self.sum_out([var], factors_containing_var[0])
        return [f for f in factors.values()]

    def compute_marginal_distribution(
        self, Q: List[str], evidence: Dict[str, bool]
    ) -> pd.DataFrame:
        """
        Computes the marginal distribution of the variables in Q given the evidence.
        """
        self.prune_network(Q, evidence)

        # reduce all cpts in network based on evidence
        for cpt in self.bn.get_all_cpts().items():
            new_cpt = self.bn.reduce_factor_rem_row(pd.Series(evidence), cpt[1])
            self.bn.update_cpt(cpt[0], new_cpt)

        ordering = self.min_degree_ordering(
            [x for x in self.bn.get_all_variables() if x not in Q]
        )
        remaining_factors = self.variable_elimination(ordering)
        # multiply remaining factors
        for _ in range(len(remaining_factors) - 1):
            first_factor = remaining_factors.pop()
            second_factor = remaining_factors.pop()
            remaining_factors.append(self.multiply_factors(first_factor, second_factor))

        joint_marginal = remaining_factors[0]

        # normalize
        joint_marginal["p"] = joint_marginal["p"] / joint_marginal["p"].sum()

        return joint_marginal

    def maximum_a_posteriori(self, Q: List[str], evidence: Dict[str, bool]):
        self.prune_network(Q, evidence)

        # reduce all cpts in network based on evidence
        for cpt in self.bn.get_all_cpts().items():
            new_cpt = self.bn.reduce_factor_rem_row(pd.Series(evidence), cpt[1])
            self.bn.update_cpt(cpt[0], new_cpt)

        ordering = self.min_degree_ordering(
            [x for x in self.bn.get_all_variables() if x not in Q]
        )
        remaining_factors = self.variable_elimination(ordering)
        # multiply remaining factors
        for _ in range(len(remaining_factors) - 1):
            first_factor = remaining_factors.pop()
            second_factor = remaining_factors.pop()
            remaining_factors.append(self.multiply_factors(first_factor, second_factor))

        joint_marginal = remaining_factors[0]
        max_row = joint_marginal.loc[joint_marginal["p"].idxmax()]
        probability = max_row["p"]
        max_row = max_row.drop("p")
        assignments = max_row.to_dict()

        return probability, assignments

    def most_probable_explanation(self, evidence: Dict[str, bool]):
        """
        Returns the most probable explanation for the given evidence.
        """

        # get all variables in network
        variables = self.bn.get_all_variables()
        # get all variables in evidence
        evidence_variables = list(evidence.keys())

        # remove edges going out from evidence variables
        graph = self.bn.structure
        for var in evidence_variables:
            graph.remove_edges_from(list(graph.out_edges(var)))

        # reduce factors in network based on evidence
        for cpt in self.bn.get_all_cpts().items():
            common_vars = [v for v in evidence.keys() if v in cpt[1].columns]
            new_cpt = cpt[1]
            for var in common_vars:
                new_cpt = new_cpt.loc[new_cpt[var] == evidence[var]]
            new_cpt = new_cpt.drop(
                columns=[v for v in common_vars if v not in new_cpt.columns[-2:]]
            )
            self.bn.update_cpt(cpt[0], new_cpt)

        ordering = self.min_degree_ordering(variables.copy())

        remaining_factors = self.variable_elimination(ordering, max_out=True)
        # multiply remaining factors
        while len(remaining_factors) > 1:
            first_factor = remaining_factors.pop()
            second_factor = remaining_factors.pop()
            remaining_factors.append(self.multiply_factors(first_factor, second_factor))

        factor = remaining_factors[0]

        return factor["p"].values[0], factor.drop("p", axis=1).to_dict("records")[0]

    def naive_prior_marginal(self, Q):
        variables = self.bn.get_all_variables()
        variables_left = set(variables)
        var = variables_left.pop()
        factor = self.bn.get_cpt(var)
        while len(variables_left) > 0:
            var = variables_left.pop()
            cpt = self.bn.get_cpt(var)
            factor = self.multiply_factors(factor, cpt)
        factor = self.sum_out([v for v in variables if v not in Q], factor)
        return factor
