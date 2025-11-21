import functools
import pickle
import sys
import time
from typing import Dict
import pandas as pd
import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gpax.evolution.tournament_selector import TournamentSelector
from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.evolution_metrics import custom_ga_metrics
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation, regression_scoring_fn, \
    regression_accuracy_evaluation_with_sgd


def run_sym_reg_ga(config: Dict):
    dataset_name = config["problem"]
    df = pd.read_csv(f"../datasets/{dataset_name}", sep=" ", header=None)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=config["seed"])

    # Create scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit only on training data, transform test data
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    key = jax.random.key(config["seed"])

    # Init the CGP policy graph with default values
    graph_structure = CGP(
        n_inputs=X.shape[1],
        n_outputs=1,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=lambda x: x,
        weighted_functions=config["solver"].get("weighted_functions", False),
        weighted_inputs=config["solver"].get("weighted_inputs", False),
        weighted_program_inputs=config["solver"].get("weighted_program_inputs", False)
    )

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config["n_pop"])
    init_cgp_genomes = jax.vmap(graph_structure.init)(keys)

    # Prepare the scoring function
    if config.get("sgd", False):
        train_fn = functools.partial(regression_accuracy_evaluation_with_sgd, graph_structure=graph_structure,
                                     X=X_train, y=y_train, reset_weights=True, batch_size=32, n_gradient_steps=100)
    else:
        train_fn = functools.partial(regression_accuracy_evaluation, graph_structure=graph_structure, X=X_train,
                                     y=y_train)
    test_fn = functools.partial(regression_accuracy_evaluation, graph_structure=graph_structure, X=X_test, y=y_test)
    scoring_fn_cgp = functools.partial(
        regression_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn
    )

    # Define a metrics function
    metrics_function = functools.partial(
        custom_ga_metrics,
        extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    # Define emitter
    cgp_mutation_fn = functools.partial(graph_structure.mutate)
    tournament_selector = TournamentSelector(tournament_size=config["tournament_size"])
    mixing_emitter = MixingEmitter(
        mutation_fn=cgp_mutation_fn,
        variation_fn=None,
        variation_percentage=0.0,  # note: CGP works with mutation only
        batch_size=config["n_offspring"],
        selector=tournament_selector
    )

    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    init_fn = functools.partial(ga.init, lamarckian=True)
    update_fn = functools.partial(ga.update, lamarckian=True)
    repertoire, emitter_state, init_metrics = init_fn(genotypes=init_cgp_genomes, population_size=config["n_pop"],
                                                      key=subkey)

    # Initialize metrics
    metrics = {key: jnp.array([]) for key in ["iteration", "max_fitness", "time", "test_accuracy"]}

    # Set up init metrics
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
                           init_metrics)
    csv_logger = CSVLogger(
        f'../results/{config["run_name"]}.csv',
        header=list(metrics.keys())
    )

    # Log initial metrics
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

    # Iterations
    for iteration in range(1, config["n_gens"]):
        start_time = time.time()
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, current_metrics = update_fn(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=subkey)
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: x[-1], current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse

        print(unwrapped_metrics)

        # Log
        csv_logger.log(unwrapped_metrics)

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses)
    )
    path = f"../results/{conf['run_name']}.pickle"
    with open(path, 'wb') as file:
        pickle.dump(repertoire_to_store, file)


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
        "sgd": True,
        "problem": "I.6.2"
    }
    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split('=')
        if key == "problem":
            conf["problem"] = value
        elif key == "seed":
            conf["seed"] = int(value)
        elif key == "sgd":
            conf["sgd"] = "t" in value

    for w_f, w_in, w_pgs in [(True, False, False), (False, True, False), (False, False, True), (False, False, False)]:
        if not (w_f * w_in * w_pgs) and conf["sgd"] == True:
            continue
        conf["solver"]["weighted_inputs"] = w_in
        conf["solver"]["weighted_functions"] = w_f
        conf["solver"]["weighted_program_inputs"] = w_pgs
        extra = "sgd" if conf["sgd"] else "std"
        extra += f"_win" if w_in else ""
        extra += f"_wfn" if w_f else ""
        extra += f"_wfn" if w_pgs else ""
        conf["run_name"] = "ga_" + conf["problem"] + "_" + extra + "_" + str(conf["seed"])
        print(conf["run_name"])
        run_sym_reg_ga(conf)
