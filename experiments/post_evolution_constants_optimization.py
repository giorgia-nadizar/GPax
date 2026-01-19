import os.path
import pickle
import sys
import time
import jax
import jax.numpy as jnp
from qdax.utils.metrics import CSVLogger

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.supervised_learning.utils import prepare_scoring_fn
from gpax.supervised_learning.dataset_utils import load_dataset


def constants_optimization_post_evolution(conf):
    if not conf["solver"]["weighted_inputs"] and not conf["solver"]["weighted_functions"]:
        if conf["constants_optimization"] != "gaussian":
            return
        for w_f, w_in, b_f, b_in in [(True, False, False, False), (False, True, False, False),
                                     (True, False, True, False), (False, True, True, False),
                                     (True, False, False, True), (False, True, False, True)]:
            config["solver"]["weighted_inputs"] = w_in
            config["solver"]["weighted_functions"] = w_f
            config["solver"]["biased_inputs"] = b_in
            config["solver"]["biased_functions"] = b_f
            extra = "none"
            extra += f"_win" if w_in else ""
            extra += f"_wfn" if w_f else ""
            extra += f"_bin" if b_in else ""
            extra += f"_bfn" if b_f else ""
            config["run_name"] = ("ga3_" + config["problem"] + "_" + extra
                                  + f"_reopt-{config['constants_reoptimization']}_" + str(config["seed"]))
            _constants_optimization_post_evolution(conf)
    else:
        _constants_optimization_post_evolution(conf)


def _constants_optimization_post_evolution(conf):
    print(conf["run_name"])
    if os.path.exists(f'../results/{conf["run_name"]}.csv'):
        print("already done")
        return
    const_optimizer = conf.get("constants_reoptimization", None)

    try:
        file = open(f"../results/{conf['repertoire_path']}.pickle", 'rb')
    except FileNotFoundError:
        return
    repertoire = pickle.load(file)

    X_train, X_test, y_train, y_test = load_dataset(conf["problem"],
                                                    scale_x=conf.get("scale_x", False),
                                                    scale_y=conf.get("scale_y", False),
                                                    random_state=conf["seed"]
                                                    )

    key = jax.random.key(conf["seed"])

    # Init the CGP policy graph with default values
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
    key1, key2 = jax.random.split(key)
    no_opt_scoring_fn_cgp = prepare_scoring_fn(X_train, y_train, X_test, y_test, graph_structure)
    scoring_fn_cgp = prepare_scoring_fn(X_train, y_train, X_test, y_test, graph_structure, const_optimizer,
                                        long_const_optimization=True)

    previous_train_accuracies, previous_extra_scores = no_opt_scoring_fn_cgp(repertoire.genotypes, key1)
    previous_best_accuracy = max(previous_train_accuracies)
    previous_best_idx = jnp.argmax(previous_train_accuracies, axis=0)
    previous_best_extra_scores = jax.tree.map(lambda x: x[previous_best_idx][0], previous_extra_scores)
    previous_best_test_accuracy = previous_best_extra_scores["test_accuracy"]

    start_time = time.time()
    train_accuracies, extra_scores = scoring_fn_cgp(repertoire.genotypes, key2)
    timelapse = time.time() - start_time
    best_accuracy = max(train_accuracies)
    best_idx = jnp.argmax(train_accuracies, axis=0)
    best_extra_scores = jax.tree.map(lambda x: x[best_idx][0], extra_scores)
    best_test_accuracy = best_extra_scores["test_accuracy"]
    best_updated_params = best_extra_scores["updated_params"]
    equation = graph_structure.get_readable_expression(best_updated_params)

    csv_logger = CSVLogger(
        f'../results/{conf["run_name"]}.csv',
        header=["iteration", "time", "max_fitness", "test_accuracy", "previous_max_fitness", "previous_test_accuracy",
                "equation"],
    )
    csv_logger.log({
        "iteration": 0,
        "time": timelapse,
        "max_fitness": best_accuracy[0],
        "test_accuracy": best_test_accuracy[0],
        "previous_max_fitness": previous_best_accuracy[0],
        "previous_test_accuracy": previous_best_test_accuracy[0],
        "equation": equation,
    }
    )
    print(best_accuracy[0], best_test_accuracy[0], previous_best_accuracy[0], previous_best_test_accuracy[0],
          equation)
    with open(f"../results/{conf['run_name']}_train_accuracies.pickle", 'wb') as file:
        pickle.dump(train_accuracies, file)
    with open(f"../results/{conf['run_name']}_extra_scores.pickle", 'wb') as file:
        pickle.dump(extra_scores, file)


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
    problems = ["chemical_2_competition", "friction_dyn_one-hot", "friction_stat_one-hot",
                "nasa_battery_1_10min",
                "nasa_battery_2_20min", "nikuradse_1", "nikuradse_2", "chemical_1_tower",
                "flow_stress_phip0.1", ]
    weights_configurations = [(True, False, False, False), (False, True, False, False),
                              (True, False, True, False), (False, True, True, False),
                              (True, False, False, True), (False, True, False, True),
                              (False, False, False, False)]

    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split('=')
        if key == "problem":
            config["problem"] = value
        elif key == "seed":
            config["seed"] = int(value)
        elif key == "constants_optimization":
            config["constants_optimization"] = value
        elif key == "problem_id":
            config["problem"] = problems[int(value)]
        elif key == "w_id":
            w_f, w_in, b_f, b_in = weights_configurations[int(value)]

    for constants_reoptimization in ["adam"]:
        # if (config["constants_optimization"] != "gaussian" and
        #         constants_reoptimization != config["constants_optimization"]):
        #     continue
        for seed in range(30):
            config["seed"] = seed
            config["constants_reoptimization"] = constants_reoptimization
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
            config["run_name"] = ("ga3_" + config["problem"] + "_" + extra
                                  + f"_reopt-{config['constants_reoptimization']}_" + str(config["seed"]))
            config["repertoire_path"] = "ga2_" + config["problem"] + "_" + extra + "_" + str(config["seed"])
            constants_optimization_post_evolution(config)
