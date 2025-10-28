import copy
import functools
import time
from copy import deepcopy
from typing import Dict

import jax
import jax.numpy as jnp

import qdax.tasks.brax.v1 as environments
from jax import tree_util
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Genotype, RNGKey
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import CSVLogger, default_ga_metrics

from gpax.cartesian_genetic_programming import CGP
from multiprocessing import Pool


def run_ga(config: Dict):
    env = environments.create(env_name=config["problem"]["env_name"],
                              episode_length=config["problem"]["episode_length"])
    reset_fn = jax.jit(env.reset)

    key = jax.random.key(config["seed"])

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        n_nodes=config["solver"]["n_nodes"],
    )

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config["n_pop"])
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)

    # Define the play step fn for CGP to interact with the env
    def cgp_play_step_fn(
            env_state,
            policy_params,
            key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_graph.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, key, transition

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[config["problem"]["env_name"]]
    scoring_fn_cgp = functools.partial(
        scoring_function,
        episode_length=config["problem"]["episode_length"],
        play_reset_fn=reset_fn,
        play_step_fn=cgp_play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    def parallel_evals_scoring_fn(genotypes: Genotype, key: RNGKey):
        n_genome_evals = config["n_genome_evals"]
        key, subkey = jax.random.split(key)
        merged_params = genotypes
        for i in range(n_genome_evals - 1):
            copied_genotypes = deepcopy(genotypes)
            merged_params = tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), merged_params, copied_genotypes)
        fitness, _, _ = scoring_fn_cgp(merged_params, subkey)
        reshaped_fitness = fitness.reshape(n_genome_evals, -1)
        averaged_fitness = jnp.mean(reshaped_fitness, axis=0).reshape(-1, 1)
        return jnp.nan_to_num(averaged_fitness, nan=-jnp.inf), {}

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[config["problem"]["env_name"]]

    # Define a metrics function
    metrics_function = functools.partial(
        default_ga_metrics,
    )

    # Define emitter
    cgp_mutation_fn = functools.partial(
        policy_graph.mutate
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=cgp_mutation_fn,
        variation_fn=None,
        variation_percentage=0.0,  # note: CGP works with mutation only
        batch_size=config["n_offspring"]
    )

    # Instantiate GA
    ga = GeneticAlgorithm(
        scoring_function=parallel_evals_scoring_fn,
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
    csv_logger = CSVLogger(
        f'results/{config["run_name"]}',
        header=list(metrics.keys())
    )

    # Log initial metrics
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

    # Iterations
    for iteration in range(1, config["n_gens"]):
        start_time = time.time()
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=subkey)
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: x[-1], current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse

        # Log
        csv_logger.log(unwrapped_metrics)


if __name__ == '__main__':
    n_cores = 36
    common_config = {
        "solver": {
            "n_nodes": 50,
        },
        "n_genome_evals": 5,
        "n_offspring": 90,
        "n_pop": 100,
        "n_gens": 3_000,
    }
    ga_hopper_config = {
                           "problem": {
                               "env_name": "hopper_uni",
                               "episode_length": 1000,
                           },
                       } | common_config
    ga_walker_config = {
                            "problem": {
                                "env_name": "walker2d_uni",
                                "episode_length": 1000,
                            },
                        } | common_config
    configs = [ga_hopper_config, ga_walker_config]
    ready_configs = []
    for seed in range(5):
        for config in configs:
            tmp_config = deepcopy(config)
            tmp_config["seed"] = seed
            tmp_config["run_name"] = f'ga_cgp_{tmp_config["problem"]["env_name"]}_{seed}'
            ready_configs.append(tmp_config)

    with Pool() as pool:
        results = pool.map(run_ga, ready_configs)

