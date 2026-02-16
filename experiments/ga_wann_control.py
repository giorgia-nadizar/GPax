import functools
import os.path
import pickle
import sys
import time
from typing import Dict, Tuple
import qdax.tasks.brax.v1 as environments
import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Genotype, RNGKey, Fitness, ExtraScores
from qdax.utils.metrics import CSVLogger, default_ga_metrics
from qdax.tasks.brax.v1.env_creators import get_mask_from_transitions

from gpax.evolution.tournament_selector import TournamentSelector
from gpax.gp.cartesian_genetic_programming import CGP


def run_rl_ga(config: Dict):
    env_name = config["problem"]
    wann_weights = jnp.asarray([-2, -1, -0.5, +0.5, +1, +2])

    env = environments.create(env_name, episode_length=1000)
    reset_fn = jax.jit(env.reset)

    key = jax.random.key(config["seed"])

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=jnp.tanh,
        weighted_functions=conf["solver"]["w_fn"],
        weighted_inputs=conf["solver"]["w_in"],
        weights_initialization="natural",
        weights_mutation=False
    )

    def cgp_play_step_fn(
            env_state,
            policy_params,
            key,
            weights_multiplier
    ):
        weights = policy_graph.get_weights(policy_params)
        conditioned_weights = jax.tree.map(lambda x: x * weights_multiplier, weights)
        actions = policy_graph.apply(policy_params, env_state.obs, conditioned_weights)

        state_desc = None  # env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

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

        return next_state, policy_params, key, transition

    def scoring_function(
            policies_params: Genotype,
            key: RNGKey, ) -> Tuple[Fitness, ExtraScores]:
        key, subkey = jax.random.split(key)
        wann_keys = jax.random.split(subkey, wann_weights.shape[0])
        vmap_single_weight_scoring_function = jax.vmap(single_weight_scoring_function, in_axes=(None, 0, 0))
        fitness, extra = vmap_single_weight_scoring_function(policies_params, wann_keys, wann_weights)
        return jnp.mean(fitness, axis=0), {}

    def single_weight_scoring_function(
            policies_params: Genotype,
            key: RNGKey,
            weight: jnp.ndarray
    ) -> Tuple[Fitness, ExtraScores]:
        # Reset environments
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, jax.tree.leaves(policies_params)[0].shape[0])
        init_states = jax.vmap(reset_fn)(keys)

        play_step_fn = functools.partial(cgp_play_step_fn, weights_multiplier=weight)

        # Step environments
        unroll_fn = functools.partial(
            generate_unroll,
            episode_length=1000,
            play_step_fn=play_step_fn,
        )
        keys = jax.random.split(key, jax.tree.leaves(policies_params)[0].shape[0])
        _, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

        # Create a mask to extract data properly
        mask = get_mask_from_transitions(data)

        # Evaluate
        fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
        # jax.debug.print("fits {x}", x=jnp.expand_dims(fitnesses, axis=1).shape)

        return jnp.expand_dims(fitnesses, axis=1), {"transitions": data}

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
    ga = GeneticAlgorithm(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
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
        print(jax.tree.map(lambda x: x[-1], metrics))

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses)
    )
    path = f"../results/{conf['run_name']}.pickle"
    with open(path, 'wb') as file:
        pickle.dump(repertoire_to_store, file)


if __name__ == '__main__':
    n_gens = 1500
    n_pop = 100
    conf = {
        "solver": {
            "n_nodes": 50,
        },
        "n_offspring": n_pop,
        "n_pop": n_pop,
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
            conf["run_name"] = f"CGP_wann_{w_txt}_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_rl_ga(conf)
