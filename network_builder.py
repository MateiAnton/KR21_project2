import itertools
import numpy as np
import pandas as pd
from BayesNet import BayesNet


def build_network(depth=3, width=2) -> BayesNet:
    bn = BayesNet()

    for i in range(depth):
        for j in range(width):
            node_name = f"{i}_{j}"
            if i > 0:
                n_edges = np.random.randint(1, width + 1)
                parents = np.random.choice(range(width), size=n_edges, replace=False)

                cpt = pd.DataFrame()
                values = list(itertools.product([False, True], repeat=n_edges + 1))
                values = list(zip(*values))
                for k, v in enumerate(values):
                    if k < len(parents):
                        cpt[f"{i-1}_{parents[k]}"] = v
                    else:
                        cpt[node_name] = v
                p = [np.random.rand() for _ in range(len(cpt) // 2)]
                p = [(i, 1 - i) for i in p]
                p = [item for sublist in p for item in sublist]
                cpt["p"] = p

                bn.add_var(node_name, cpt)
                for parent in parents:
                    bn.add_edge((f"{i-1}_{parent}", node_name))
            else:
                p = np.random.rand()
                cpt = pd.DataFrame(data={node_name: [True, False], "p": [p, 1 - p]})
                bn.add_var(node_name, cpt)

    return bn
