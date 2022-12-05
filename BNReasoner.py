from typing import Union
from BayesNet import BayesNet
from typing import List, Dict
import pandas as pd


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

    # TODO: This is where your methods should go
    def prune_leafs_rec(self, target_vars: List[str]):
        leafs = [node for node in self.bn.structure.nodes() if self.bn.structure.out_degree(node) == 0]
        leafs_to_del = [leaf for leaf in leafs if leaf not in target_vars]
        for leaf in leafs_to_del:
            self.bn.del_var(leaf)
        if len(leafs_to_del) != 0:
            self.prune_leafs_rec(target_vars)

    def prune_network(self, vars: List[str], e: Dict[str, bool]):
        if not all(var in self.bn.get_all_variables() for var in vars):
            raise Exception('provided variables not in network')
        if not all(key in self.bn.get_all_variables() for key in e.keys()):
            raise Exception('provided evidence not part of netowrk')

        for var in e.keys():
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))
                cpt = self.bn.get_cpt(child)
                e_without_factor = e.copy()
                if child in e:
                    del e_without_factor[child]
                cpt = self.bn.reduce_factor_rem_row(pd.Series(e_without_factor), cpt)
                print(cpt)
                cpt = cpt.drop(columns = [var])
                # cpt = cpt[cpt['p'] != 0]
                self.bn.update_cpt(child, cpt)
                
        self.prune_leafs_rec(list(set(vars).union(set(e.keys()))))
