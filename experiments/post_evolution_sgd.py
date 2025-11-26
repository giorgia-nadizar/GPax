import functools
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from qdax.utils.metrics import CSVLogger
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation_with_sgd, \
    regression_accuracy_evaluation, regression_scoring_fn


def sgd_post_evolution(conf):
    file = open(f"../results/{conf['run_name']}.pickle", 'rb')
    repertoire = pickle.load(file)
    int_genotypes = jax.tree.map(lambda x : x.astype(int), repertoire.genotypes)

    X, y = load_diabetes(return_X_y=True)

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
        outputs_wrapper=lambda x: x,
        weighted_functions=conf["solver"].get("weighted_functions", False),
        weighted_inputs=conf["solver"].get("weighted_inputs", False),
        weighted_program_inputs=conf["solver"].get("weighted_program_inputs", False),
        weights_mutation=False
    )
    if conf.get("weights") is not None:
        train_fn = functools.partial(regression_accuracy_evaluation_with_sgd, graph_structure=graph_structure,
                                     X=X_train, y=y_train, reset_weights=True, optimizer=optax.lbfgs())
    else:
        train_fn = functools.partial(regression_accuracy_evaluation_with_sgd, graph_structure=graph_structure,
                                 X=X_train, y=y_train, reset_weights=True, batch_size=32, n_gradient_steps=10_000)
    test_fn = functools.partial(regression_accuracy_evaluation, graph_structure=graph_structure, X=X_test, y=y_test)
    scoring_fn_cgp = functools.partial(
        regression_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn
    )
    start_time = time.time()
    train_accuracies, extra_scores = scoring_fn_cgp(int_genotypes, key)
    timelapse = time.time() - start_time
    best_accuracy = max(train_accuracies)
    best_idx = jnp.argmax(train_accuracies, axis=0)
    best_extra_scores = jax.tree.map(lambda x: x[best_idx], extra_scores)
    best_test_accuracy = best_extra_scores["test_accuracy"]

    csv_logger = CSVLogger(
        f'../results/{conf["run_name"]}_{conf["weights"]}.csv',
        header=["iteration", "time", "max_fitness", "test_accuracy"]
    )
    csv_logger.log({
        "iteration": conf["n_gens"],
        "time": timelapse,
        "max_fitness": best_accuracy[0],
        "test_accuracy": best_test_accuracy[0][0]
    }
    )
    print(best_accuracy[0], best_test_accuracy[0][0])


if __name__ == '__main__':
    conf = {
        "solver": {
            "n_nodes": 50,
        },
        "n_offspring": 90,
        "n_pop": 100,
        "n_gens": 5_000,
        "seed": 0,
        "tournament_size": 3,
        "problem": "diabetes",
        "sgd": False,
    }
    for seed in range(5):
        conf["seed"] = seed
        for w_f, w_in, w_pgs in [(True, False, False), (False, True, False), (False, False, True)]:
            # TODO add the re-optimization even if the weights were evolved with mutation
            conf["solver"]["weighted_inputs"] = w_in
            conf["solver"]["weighted_functions"] = w_f
            conf["solver"]["weighted_program_inputs"] = w_pgs
            conf["weights"] = "win" if w_in else ("wfn" if w_f else "wpgs") + "-bfgs"
            # extra = "sgd" if conf["sgd"] else "std"
            # extra += f"_win" if w_in else ""
            # extra += f"_wfn" if w_f else ""
            # extra += f"_wpgs" if w_pgs else ""
            extra = "std"
            conf["run_name"] = "ga_" + conf["problem"] + "_" + extra + "_" + str(conf["seed"])
            print(conf["run_name"], conf["weights"])
            sgd_post_evolution(conf)
