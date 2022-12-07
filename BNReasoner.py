from typing import Union
from BayesNet import BayesNet
from typing import List, Dict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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
        merged = pd.merge(
            factor1.drop(columns=["p"]), factor2.drop(columns=["p"]), how="outer"
        )
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
            nx.draw(interaction_graph, with_labels=True, node_size=3000)
            plt.show()

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

    def sum_out(self, var: str, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Sums out a variable from a factor and returns the resulting factor.
        """
        df = factor.drop(columns=[var])
        return df.groupby(df.columns.drop("p").tolist(), as_index=False).sum()

    def max_out(self, var: str, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Maxes out a variable from a factor and returns the resulting factor.
        """
        return factor.sort_values(by=["p"], ascending=False).drop_duplicates(
            factor.columns.drop(["p", var])
        )

    def variable_elimination(self, factor: pd.DataFrame, variable: str) -> pd.DataFrame:
        """
        Sums out the variable in the factor and returns the resulting factor.
        """
        cols = list(factor.columns)
        cols.remove(variable)
        cols.remove("p")
        return factor.groupby(by=cols, as_index=False).sum().drop(columns=[variable])

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

        print(self.bn.get_all_cpts())
        ordering = self.min_degree_ordering(
            [x for x in self.bn.get_all_variables() if x not in Q]
        )
        print(ordering)
        curr_factor = 1
        print(curr_factor)
        for var in ordering:
            factor = self.bn.get_cpt(var)
            if isinstance(curr_factor, int):
                curr_factor = factor
            print(" ")
            print(curr_factor)
            for parent in self.bn.get_parents(var):
                print(self.bn.get_cpt(parent))
                curr_factor = self.multiply_factors(
                    curr_factor, self.bn.get_cpt(parent)
                )
            print(" ")
            print(curr_factor)
            print(var)
            curr_factor = self.variable_elimination(curr_factor, var)
        return curr_factor
