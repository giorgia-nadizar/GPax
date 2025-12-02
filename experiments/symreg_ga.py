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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gpax.evolution.tournament_selector import TournamentSelector
from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.evolution_metrics import custom_ga_metrics
from gpax.symbolicregression.utils import prepare_scoring_fn


def run_sym_reg_ga(config: Dict):
    # dataset_name = config["problem"]
    # df = pd.read_csv(f"../datasets/{dataset_name}", sep=" ", header=None)
    # X = df.iloc[:, :-1].to_numpy()
    # y = df.iloc[:, -1].to_numpy()
    const_optimizer = config.get("constants_optimization", None)

    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=config["seed"])

    # Create scalers
    if config.get("scale_x", False):
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
    if config.get("scale_y", False):
        y_scaler = StandardScaler()
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
        weighted_program_inputs=config["solver"].get("weighted_program_inputs", False),
        weights_mutation=const_optimizer in [False, "automl0"],
        weights_mutation_type="automl0" if const_optimizer == "automl0" else "gaussian"
    )
    print(graph_structure)

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config["n_pop"])
    init_cgp_genomes = jax.vmap(graph_structure.init)(keys)

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

    # Prepare the scoring function
    scoring_fn_cgp = prepare_scoring_fn(X_train, y_train, X_test, y_test, graph_structure, const_optimizer)
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=True
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(genotypes=init_cgp_genomes, population_size=config["n_pop"],
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

        # TODO
        # update the current dataset sample
        # update the scoring fn
        # update the ga

        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
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
        "constants_optimization": "automl0",
        "problem": "I.6.2"
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

    # for problem in ["I.13.12","I.6.2","II.24.17"]:
    for w_f, w_in, w_pgs in [(True, False, False), (False, True, False), (False, False, True), (False, False, False)]:
        if not (w_f or w_in or w_pgs) and conf["sgd"] == True:
            continue
        conf["problem"] = "diabetes"
        conf["solver"]["weighted_inputs"] = w_in
        conf["solver"]["weighted_functions"] = w_f
        conf["solver"]["weighted_program_inputs"] = w_pgs
        extra = conf["constants_optimization"] if conf["constants_optimization"] else "std"
        extra += f"_win" if w_in else ""
        extra += f"_wfn" if w_f else ""
        extra += f"_wpgs" if w_pgs else ""
        conf["run_name"] = "ga_" + conf["problem"] + "_" + extra + "_" + str(conf["seed"])
        print(conf["run_name"])
        run_sym_reg_ga(conf)
