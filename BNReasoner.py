from typing import Union
from BayesNet import BayesNet
from typing import List, Dict
import pandas as pd
import networkx as nx


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
