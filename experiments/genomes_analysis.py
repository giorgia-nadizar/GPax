import pickle
import sys
from collections import Counter
from typing import Dict
import jax
import jax.numpy as jnp
import pandas as pd

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.supervised_learning.dataset_utils import load_dataset


def analyze_genome(conf: Dict) -> Dict:
    try:
        file = open(f"../results/{conf['repertoire_path']}.pickle", 'rb')
    except FileNotFoundError:
        return {}
    repertoire = pickle.load(file)

    X_train, X_test, y_train, y_test = load_dataset(conf["problem"],
                                                    scale_x=conf.get("scale_x", False),
                                                    scale_y=conf.get("scale_y", False),
                                                    random_state=conf["seed"]
                                                    )

    graph_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=1,
        n_nodes=conf["solver"]["n_nodes"],
        n_input_constants=conf["solver"]["n_input_constants"],
        outputs_wrapper=lambda x: x,
        weighted_functions=conf["solver"].get("weighted_functions", False),
        weighted_inputs=conf["solver"].get("weighted_inputs", False),
        weighted_program_inputs=conf["solver"].get("weighted_program_inputs", False),
        weights_mutation=False,
    )

    best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
    best_genotype = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)

    active_mask = graph_structure.compute_active_mask(best_genotype)
    functions = best_genotype["genes"]["functions"].astype(int)[0][active_mask == 1]
    connections_1 = best_genotype["genes"]["inputs1"].astype(int)[0][active_mask == 1]
    connections_2 = best_genotype["genes"]["inputs2"].astype(int)[0][active_mask == 1]

    # find which inputs are used
    n_used_inputs = 0
    for i in range(graph_structure.n_inputs):
        if i in connections_1 or i in connections_2:
            n_used_inputs += 1

    # functions
    function_set = graph_structure.function_set
    counts = Counter(functions.tolist())
    info_dict = {f"n_fn_{i}": counts.get(i, 0) for i in range(len(function_set))}
    info_dict["used_inputs_fraction"] = n_used_inputs / graph_structure.n_inputs
    info_dict["active_fraction"] = jnp.sum(active_mask) / len(active_mask)

    n_one_arity = 0
    n_two_arity = 0
    for fn in functions.tolist():
        if function_set.arities[fn] == 1:
            n_one_arity += 1
        else:
            n_two_arity += 1

    info_dict["n_one_arity"] = n_one_arity
    info_dict["n_two_arity"] = n_two_arity
    info_dict["run_name"] = conf['repertoire_path']

    return info_dict


if __name__ == '__main__':
    config = {
        "solver": {
            "n_nodes": 100,
            "n_input_constants": 2,
            "weights_initialization": "uniform"
        },
        "n_offspring": 90,
        "n_pop": 100,
        "seed": 0,
        "tournament_size": 3,
        "problem": "feynman_I_6_2",
        "scale_x": False,
        "scale_y": False,
        "constants_optimization": "gaussian",
        "constants_reoptimization": "adam",
    }

    info_dicts = []

    for seed in range(5):
        for constants_optimization in ["adam", "gaussian"]:
            for w_f, w_in, b_f, b_in in [(True, False, False, False), (False, True, False, False),
                                         (True, False, True, False), (False, True, True, False),
                                         (True, False, False, True), (False, True, False, True),
                                         (False, False, False, False)]:
                for problem in ["chemical_2_competition", "friction_dyn_one-hot", "friction_stat_one-hot",
                                "nasa_battery_1_10min",
                                "nasa_battery_2_20min", "nikuradse_1", "nikuradse_2", "chemical_1_tower",
                                "flow_stress_phip0.1", ]:
                    config["problem"] = problem
                    config["constants_optimization"] = constants_optimization
                    config["seed"] = seed
                    config["solver"]["weighted_inputs"] = w_in
                    config["solver"]["weighted_functions"] = w_f
                    config["solver"]["weighted_program_inputs"] = False
                    config["solver"]["biased_inputs"] = b_in
                    config["solver"]["biased_functions"] = b_f
                    extra = config["constants_optimization"]
                    extra += f"_win" if w_in else ""
                    extra += f"_wfn" if w_f else ""
                    extra += f"_bin" if b_in else ""
                    extra += f"_bfn" if b_f else ""
                    extra += "_n" if config["solver"].get("weights_initialization") == "ones" else ""
                    config["repertoire_path"] = "ga2_" + config["problem"] + "_" + extra + "_" + str(config["seed"])
                    info_dict = analyze_genome(config)
                    info_dict["seed"] = seed
                    info_dict["w"] = extra.replace(constants_optimization, "")
                    info_dict["opt"] = constants_optimization
                    info_dict["problem"] = problem
                    info_dicts.append(info_dict)
                    print(config["repertoire_path"])

    info_df = pd.DataFrame(info_dicts)
    info_df.to_csv("../results/genomes_analysis.csv", index=False)
