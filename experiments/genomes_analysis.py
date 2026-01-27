import pickle
import sys
from collections import Counter
from typing import Dict
import jax
import jax.numpy as jnp
import pandas as pd

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.graphs.functions import FunctionSet, JaxFunction
from gpax.supervised_learning.dataset_utils import load_dataset

eps = 1e-6
function_set_small = {
    "plus": JaxFunction(lambda x, y: x + y, 2, "+"),
    "minus": JaxFunction(lambda x, y: x - y, 2, "-"),
    "times": JaxFunction(lambda x, y: x * y, 2, "*"),
    "div": JaxFunction(lambda x, y: x / y, 2, "/"),
}

function_set_trig = {
    "plus": JaxFunction(lambda x, y: x + y, 2, "+"),
    "minus": JaxFunction(lambda x, y: x - y, 2, "-"),
    "times": JaxFunction(lambda x, y: x * y, 2, "*"),
    "div": JaxFunction(lambda x, y: x / y, 2, "/"),
    "sin": JaxFunction(lambda x, y: jnp.sin(x), 1, "sin"),
    "cos": JaxFunction(lambda x, y: jnp.cos(x), 1, "cos"),
}


def analyze_genome(conf: Dict) -> Dict:
    try:
        file = open(f"../results/{conf['repertoire_path']}.pickle", 'rb')
    except FileNotFoundError:
        print(f"../results/{conf['repertoire_path']}.pickle")
        return {}
    repertoire = pickle.load(file)

    X_train, X_test, y_train, y_test = load_dataset(conf["problem"],
                                                    scale_x=conf.get("scale_x", False),
                                                    scale_y=conf.get("scale_y", False),
                                                    random_state=conf["seed"]
                                                    )

    functions_dict = function_set_trig if ("4" in conf["problem"] or "7" in conf["problem"]) else function_set_small
    function_set = FunctionSet(functions_dict=functions_dict)

    # Init the CGP policy graph with default values
    graph_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=1,
        function_set=function_set,
        n_nodes=conf["solver"]["n_nodes"],
        n_input_constants=conf["solver"]["n_input_constants"],
        outputs_wrapper=lambda x: x,
        weighted_functions=conf["solver"].get("weighted_functions", False),
        weighted_inputs=conf["solver"].get("weighted_inputs", False),
        weighted_program_inputs=conf["solver"].get("weighted_program_inputs", False),
        weights_mutation=False,
        weights_mutation_type="gaussian",
        weights_initialization=conf["solver"].get("weights_initialization", "uniform"),
    )

    best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
    best_genotype = jax.tree.map(lambda x: x[best_idx][0], repertoire.genotypes)

    info_dict = {}
    info_dict["best_expression"] = graph_structure.get_readable_expression(best_genotype, inputs_mapping={0: "x"},
                                                                           outputs_mapping={0: "y"})
    info_dict["run_name"] = conf['repertoire_path']

    return info_dict


if __name__ == '__main__':
    config = {
        "solver": {
            "n_nodes": 15,
            "n_input_constants": 0,
            "weights_initialization": "uniform"
        },
        "n_offspring": 8,
        "n_pop": 10,
        "seed": 0,
        "tournament_size": 3,
        "problem": "dcgp_1",
        "scale_x": False,
        "scale_y": False,
        "constants_optimization": "adam",
    }

    for problem in [f"dcgp_{i}" for i in range(1, 8)]:
        print(problem)
        info_dicts = []
        for seed in range(30):
            print(seed)
            for constants_optimization in ["adam", "gaussian"]:
                for w_f, w_in, b_f, b_in in [(True, False, False, False), (False, True, False, False),
                                             (True, False, True, False), (False, True, True, False),
                                             (True, False, False, True), (False, True, False, True),
                                             (False, False, False, False)]:
                    if not (w_f or w_in) and constants_optimization == "adam":
                        continue
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
                    # print(config["repertoire_path"])

        info_df = pd.DataFrame(info_dicts)
        info_df.to_csv(f"../results/genomes_analysis_{problem}.csv", index=False)
