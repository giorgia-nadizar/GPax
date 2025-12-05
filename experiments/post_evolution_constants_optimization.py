import pickle
import sys
import time
import pandas as pd
import jax
import jax.numpy as jnp
from qdax.utils.metrics import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.utils import prepare_scoring_fn


def constants_optimization_post_evolution(conf):
    const_optimizer = conf.get("constants_reoptimization", None)

    file = open(f"../results/{conf['repertoire_path']}.pickle", 'rb')
    repertoire = pickle.load(file)

    dataset_name = conf["problem"]
    df = pd.read_csv(f"../datasets/{dataset_name}.tsv", sep="\t")
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=conf["seed"])

    # Create scalers
    if conf.get("scale_x", False):
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
    if conf.get("scale_y", False):
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

    key = jax.random.key(conf["seed"])

    # Init the CGP policy graph with default values
    graph_structure = CGP(
        n_inputs=X.shape[1],
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
    previous_best_extra_scores = jax.tree.map(lambda x: x[previous_best_idx], previous_extra_scores)
    previous_best_test_accuracy = previous_best_extra_scores["test_accuracy"]

    start_time = time.time()
    train_accuracies, extra_scores = scoring_fn_cgp(repertoire.genotypes, key2)
    timelapse = time.time() - start_time
    best_accuracy = max(train_accuracies)
    best_idx = jnp.argmax(train_accuracies, axis=0)
    best_extra_scores = jax.tree.map(lambda x: x[best_idx], extra_scores)
    best_test_accuracy = best_extra_scores["test_accuracy"]

    csv_logger = CSVLogger(
        f'../results/{conf["run_name"]}.csv',
        header=["iteration", "time", "max_fitness", "test_accuracy", "previous_max_fitness", "previous_test_accuracy"],
    )
    csv_logger.log({
        "iteration": 0,
        "time": timelapse,
        "max_fitness": best_accuracy[0],
        "test_accuracy": best_test_accuracy[0][0],
        "previous_max_fitness": previous_best_accuracy[0],
        "previous_test_accuracy": previous_best_test_accuracy[0][0],
    }
    )
    print(best_accuracy[0], best_test_accuracy[0][0], previous_best_accuracy[0], previous_best_test_accuracy[0][0])


if __name__ == '__main__':
    conf = {
        "solver": {
            "n_nodes": 50,
            "n_input_constants": 5
        },
        "n_offspring": 90,
        "n_pop": 100,
        "seed": 0,
        "tournament_size": 3,
        "problem": "feynman_I_6_2",
        "scale_x": False,
        "scale_y": True,
        "constants_optimization": "gaussian",
        "constants_reoptimization": "adam",
    }
    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split('=')
        if key == "problem":
            conf["problem"] = value
        elif key == "seed":
            conf["seed"] = int(value)
        elif key == "constants_optimization":
            conf["constants_optimization"] = value

    for constants_reoptimization in ["adam", "cmaes"]:
        for problem in ["I_13_12", "I_6_2", "II_24_17"]:
            conf["problem"] = f"feynman_{problem}"
            conf["constants_reoptimization"] = constants_reoptimization
            for w_f, w_in, w_pgs in [(True, False, False), (False, True, False), (False, False, True)]:
                conf["solver"]["weighted_inputs"] = w_in
                conf["solver"]["weighted_functions"] = w_f
                conf["solver"]["weighted_program_inputs"] = w_pgs
                extra = conf['constants_optimization']
                extra += "_win" if w_in else ""
                extra += "_wfn" if w_f else ""
                extra += "_wpgs" if w_pgs else ""
                conf["run_name"] = ("ga_" + conf["problem"] + "_" + extra
                                    + f"_reopt-{conf['constants_reoptimization']}_" + str(conf["seed"]))
                conf["repertoire_path"] = "ga_" + conf["problem"] + "_" + extra + "_" + str(conf["seed"])
                print(conf["run_name"])
                constants_optimization_post_evolution(conf)
