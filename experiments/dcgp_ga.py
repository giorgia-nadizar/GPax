import functools
import os.path
import pickle
import sys
import time
from typing import Dict, List

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger

from gpax.evolution.tournament_selector import TournamentSelector
from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.evolution_metrics import custom_ga_metrics
from gpax.graphs.functions import JaxFunction, FunctionSet
from gpax.supervised_learning.dataset_utils import downsample_dataset, load_dataset
from gpax.supervised_learning.utils import prepare_scoring_fn, prepare_rescoring_fn

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


def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
    test_accuracy_values = metrics.pop("test_accuracy")
    for idx, header in enumerate(headers):
        metrics[header] = test_accuracy_values[idx]
    return metrics


def run_sym_reg_ga(config: Dict):
    const_optimizer = config.get("constants_optimization", None)
    task = "regression" if "mtr" not in config["problem"] else "multiregression"

    X_train, X_test, y_train, y_test = load_dataset(config["problem"],
                                                    scale_x=config.get("scale_x", False),
                                                    scale_y=config.get("scale_y", False),
                                                    random_state=config["seed"]
                                                    )
    key = jax.random.key(config["seed"])
    sample_key, key = jax.random.split(key)
    rescoring = (len(X_train) > 2048)

    # adjust gens if constants optimization
    # if const_optimizer == "adam":
    #     config["n_gens"] = int((config["n_gens"] * min(2048, len(X_train))) / (min(2048, len(X_train)) + 3200))

    if rescoring:
        downsample_fn = functools.partial(downsample_dataset, size=config.get("dataset_size", 1024))
        X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
    else:
        X_train_sub, y_train_sub = X_train, y_train

    functions_dict = function_set_trig if ("4" in config["problem"] or "7" in config["problem"]) else function_set_small
    function_set = FunctionSet(functions_dict=functions_dict)

    # Init the CGP policy graph with default values
    graph_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=1,
        function_set=function_set,
        n_nodes=config["solver"]["n_nodes"],
        n_input_constants=config["solver"]["n_input_constants"],
        outputs_wrapper=lambda x: x,
        weighted_functions=config["solver"].get("weighted_functions", False),
        weighted_inputs=config["solver"].get("weighted_inputs", False),
        weighted_program_inputs=config["solver"].get("weighted_program_inputs", False),
        weights_mutation=const_optimizer in ["gaussian", "automl0"],
        weights_mutation_type="automl0" if const_optimizer == "automl0" else "gaussian",
        weights_initialization=config["solver"].get("weights_initialization", "uniform"),
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
    scoring_fn_cgp = prepare_scoring_fn(X_train_sub, y_train_sub, X_test, y_test, graph_structure, const_optimizer,
                                        task=task)
    rescoring_fn_cgp = prepare_rescoring_fn(X_train_sub, y_train_sub, graph_structure, task=task)
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=True,
        rescoring_function=rescoring_fn_cgp
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(genotypes=init_cgp_genomes, population_size=config["n_pop"],
                                                      key=subkey)

    # Initialize metrics
    n_targets = y_test.shape[1]
    test_accuracy_header = ["test_accuracy"] if n_targets == 1 else [f"rrmse_{i}" for i in range(n_targets)]
    metrics = {key: jnp.array([]) for key in ["iteration", "max_fitness", "time"] + test_accuracy_header}

    # Set up init metrics
    # init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = 0
    init_metrics["max_fitness"] = init_metrics["max_fitness"][0]
    init_metrics["time"] = 0.0  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    # metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
    #                        init_metrics)
    csv_logger = CSVLogger(
        f'../results/{config["run_name"]}.csv',
        header=list(metrics.keys())
    )

    # Log initial metrics
    # csv_logger.log(jax.tree.map(lambda x: x[-1], init_metrics))
    csv_logger.log(process_metrics_mtr(init_metrics, test_accuracy_header))

    # Iterations
    for iteration in range(1, config["n_gens"]):
        key, subkey, sample_key = jax.random.split(key, 3)

        if rescoring:
            # change batch of the dataset to evaluate upon
            X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
            scoring_fn_cgp = prepare_scoring_fn(X_train_sub, y_train_sub, X_test, y_test, graph_structure,
                                                const_optimizer, task=task)
            rescoring_fn_cgp = prepare_rescoring_fn(X_train_sub, y_train_sub, graph_structure, task=task)
            ga = ga.replace_scoring_fns(scoring_fn_cgp, rescoring_fn_cgp)

        start_time = time.time()
        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=subkey, rescore_repertoire=rescoring)
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: jnp.ravel(x), current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse
        unwrapped_metrics["max_fitness"] = unwrapped_metrics["max_fitness"][0]
        if len(test_accuracy_header) > 1:
            unwrapped_metrics = process_metrics_mtr(unwrapped_metrics, test_accuracy_header)

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
    n_gens = 10000
    conf = {
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
        "constants_optimization": "gaussian",
    }

    problems = [f"dcgp_{i + 1}" for i in range(7)]

    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split('=')
        if key == "problem":
            conf["problem"] = value
        elif key == "seed":
            conf["seed"] = int(value)
        elif key == "problem_id":
            conf["problem"] = problems[int(value)]
        elif key == "constants_optimization":
            conf["constants_optimization"] = value

    for seed in range(30):
        conf["seed"] = seed
        for w_f, w_in, b_f, b_in in [(True, False, False, False), (False, True, False, False),
                                     (True, False, True, False), (False, True, True, False),
                                     (True, False, False, True), (False, True, False, True),
                                     (False, False, False, False)]:
            if not (w_f or w_in) and conf["constants_optimization"] not in ["mutation", "automl0", "gaussian"]:
                continue
            conf["solver"]["weighted_inputs"] = w_in
            conf["solver"]["weighted_functions"] = w_f
            conf["solver"]["weighted_program_inputs"] = False
            conf["solver"]["biased_inputs"] = b_in
            conf["solver"]["biased_functions"] = b_f
            conf["n_gens"] = n_gens
            extra = conf["constants_optimization"]
            extra += f"_win" if w_in else ""
            extra += f"_wfn" if w_f else ""
            extra += f"_bin" if b_in else ""
            extra += f"_bfn" if b_f else ""
            # extra += f"_wpgs" if w_pgs else ""
            extra += "_n" if conf["solver"].get("weights_initialization") == "natural" else ""
            conf["run_name"] = "ga2_" + conf["problem"].replace("/", "_") + "_" + extra + "_" + str(conf["seed"])
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_sym_reg_ga(conf)
