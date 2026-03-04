import functools
import os.path
import pickle
import sys
import time
from typing import Dict, Tuple
import qdax.tasks.brax.v1 as environments
import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
from jax.flatten_util import ravel_pytree
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Genotype, RNGKey, Fitness, ExtraScores
from qdax.utils.metrics import CSVLogger, default_ga_metrics
from qdax.tasks.brax.v1.env_creators import get_mask_from_transitions

from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.tournament_selector import TournamentSelector
from gpax.gp.cartesian_genetic_programming import CGP


def opt_single_genome_openai_es(genome: Genotype, rnd_key: RNGKey, config: Dict) -> Tuple[Genotype, ExtraScores]:
    env_name = config["problem"]

    num_iterations = conf["n_inner_gens"]
    population_size = conf["n_inner_pop"]

    # Create training environment
    env = environments.create(
        env_name=env_name,
        episode_length=1000,
    )

    # Init the CGP policy graph with default values
    policy_graph_structure = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        weighted_functions=config["solver"]["w_fn"],
        weighted_inputs=config["solver"]["w_in"],
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=jnp.tanh
    )

    init_weights = policy_graph_structure.get_weights(genome)
    weights_array_sample, weights_tree_def = ravel_pytree(init_weights)

    solver = Evosax2JAX_Wrapper(
        Strategies["OpenES"],
        param_size=weights_array_sample.shape[0],
        pop_size=population_size,
        es_config={"maximize": True,
                   "centered_rank": True,
                   "lrate_init": 0.01,
                   "lrate_decay": 0.999,
                   "lrate_limit": 0.001},
        es_params={"sigma_init": 0.05,
                   "sigma_decay": 0.999,
                   "sigma_limit": 0.01},
        seed=0,
    )
    evolutionary_strategy = solver.es
    es_state = solver.es_state

    reset_key, rnd_key = jax.random.split(rnd_key)
    env_init_state = env.reset(reset_key)

    def _single_unroll(genotype_weights_array: jnp.ndarray) -> QDTransition:
        genotype_weights_dict = weights_tree_def(genotype_weights_array)

        def _cgp_play_step_fn(
                env_state, x
        ):
            actions = policy_graph_structure.apply(genome, env_state.obs, genotype_weights_dict)
            next_state = env.step(env_state, actions)

            state_desc = None

            transition = QDTransition(
                obs=env_state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                dones=next_state.done,
                actions=actions,
                truncations=next_state.info["truncation"],
                state_desc=state_desc,
                next_state_desc=None  # next_state.info["state_descriptor"],
            )

            return next_state, transition

        _, transitions = jax.lax.scan(
            _cgp_play_step_fn,
            env_init_state,
            (),
            length=1000,
        )

        return transitions

    vmapped_unroll_fn = jax.jit(jax.vmap(_single_unroll))

    def fitness_fn(genotypes_weights_arrays: jnp.ndarray) -> jnp.ndarray:
        transitions = vmapped_unroll_fn(genotypes_weights_arrays)
        mask = get_mask_from_transitions(transitions)
        fitnesses = jnp.sum(transitions.rewards * (1.0 - mask), axis=1)
        return fitnesses

    def opt_loop(carry, x):
        inner_es_state, inner_key = carry
        inner_key, ask_key = jax.random.split(inner_key)
        samples, inner_es_state = evolutionary_strategy.ask(ask_key, inner_es_state)
        scores = fitness_fn(samples)
        inner_es_state = evolutionary_strategy.tell(samples, scores, inner_es_state)
        return (inner_es_state, inner_key), x

    (es_state, _), _ = jax.lax.scan(
        opt_loop,
        (es_state, rnd_key),
        (),
        length=num_iterations,
    )
    best_score = es_state.best_fitness
    best_params_array = es_state.best_member

    best_params_pytree = weights_tree_def(best_params_array)
    updated_params = policy_graph_structure.update_weights(genome, best_params_pytree)
    return best_score, {"updated_params": updated_params}


def run_rl_ga(config: Dict):
    env_name = config["problem"]

    env = environments.create(env_name, episode_length=1000)
    key = jax.random.key(config["seed"])

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=jnp.tanh,
        weighted_functions=conf["solver"]["w_fn"],
        weighted_inputs=conf["solver"]["w_in"],
        weights_initialization="uniform",
        weights_mutation=False
    )
    constants_opt_fn = functools.partial(opt_single_genome_openai_es, config=config)

    def scoring_function(
            policies_params: Genotype,
            key: RNGKey,
    ) -> Tuple[Fitness, ExtraScores]:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, jax.tree.leaves(policies_params)[0].shape[0])
        fitnesses, updated_params = jax.jit(jax.vmap(constants_opt_fn,in_axes=(0,0)))(policies_params, keys)
        return jnp.expand_dims(fitnesses, axis=1), updated_params

    # Init the population of trees
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config["n_pop"])
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)

    # Define a metrics function
    metrics_function = functools.partial(default_ga_metrics)

    # Define emitter
    cgp_mutation_fn = functools.partial(
        policy_graph.mutate  # , mutation_probabilities={"inputs" : .2}
    )
    tournament_selector = TournamentSelector(tournament_size=config["tournament_size"])
    mixing_emitter = MixingEmitter(
        mutation_fn=cgp_mutation_fn,
        variation_fn=None,
        variation_percentage=0.0,
        batch_size=config["n_offspring"],
        selector=tournament_selector
    )

    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=True,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(genotypes=init_cgp_genomes, population_size=config["n_pop"],
                                                      key=subkey)

    # Initialize metrics
    metrics = {key: jnp.array([]) for key in ["iteration", "max_fitness", "time"]}

    # Set up init metrics
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
                           init_metrics)
    print(metrics)

    csv_logger = CSVLogger(
        f'../results/{config["run_name"]}.csv',
        header=list(metrics.keys())
    )

    # Log initial metrics
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

    log_period = 10
    num_loops = config["n_gens"] // log_period

    # Iterations
    ga_scan_update = ga.scan_update
    for i in range(num_loops):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            ga_scan_update,
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        current_metrics["iteration"] = jnp.arange(1 + log_period * i, 1 + log_period * (i + 1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)

        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric.ravel()], axis=0),
                               metrics, current_metrics)

        # Log
        csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses)
    )
    path = f"../results/{conf['run_name']}.pickle"
    with open(path, 'wb') as file:
        pickle.dump(repertoire_to_store, file)


if __name__ == '__main__':
    n_gens = 500
    n_pop = 50
    conf = {
        "solver": {
            "n_nodes": 50,
        },
        "n_offspring": n_pop,
        "n_pop": n_pop,
        "n_inner_gens": 30,
        "n_inner_pop": 30,
        "seed": 0,
        "tournament_size": 3,
        "problem": "hopper",
    }

    tasks = ["hopper", "walker2d", "halfcheetah", "swimmer", "reacher", ]

    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split("=")
        if key == "problem_id":
            conf["problem"] = tasks[int(value)]

    for w in [True, False]:
        for seed in range(10):
            conf["solver"]["w_fn"] = w
            conf["solver"]["w_in"] = not w
            w_txt = "w_fn" if w else "w_in"
            conf["seed"] = seed
            conf["n_gens"] = n_gens
            conf["run_name"] = f"CGP+ES_{w_txt}_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_rl_ga(conf)
