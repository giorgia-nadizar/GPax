import pickle
import sys
import time
from functools import partial
from typing import Dict
from typing import Tuple

import jax
import jax.numpy as jnp
import pandas as pd
import qdax.tasks.brax.v1 as environments
from jax.flatten_util import ravel_pytree
from qdax.baselines.cmaes import CMAES
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Genotype
from qdax.tasks.brax.v1.env_creators import get_mask_from_transitions
from qdax.utils.metrics import CSVLogger

from gpax.gp.cartesian_genetic_programming import CGP


def analyze_policy_behavior(genome: Genotype, config: Dict) -> Tuple[jnp.ndarray, float]:
    env_name = config["problem"]

    # Create training environment
    env = environments.create(
        env_name=env_name,
        episode_length=1000,
    )

    key = jax.random.key(config["seed"])
    reset_key, key = jax.random.split(key)
    env_init_state = env.reset(reset_key)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        weighted_functions=config["solver"]["w_fn"],
        weighted_inputs=config["solver"]["w_in"],
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=jnp.tanh
    )

    init_weights = policy_graph.get_weights(genome)

    cum_reward = 0
    actions_tracking = []
    env_state = env_init_state
    for _ in range(1000):
        actions = policy_graph.apply(genome, env_state.obs, init_weights)
        actions_tracking.append(actions)
        env_state = env.step(env_state, actions)
        if env_state.done:
            break
        cum_reward += env_state.reward

    return jnp.vstack(actions_tracking), cum_reward


def run_analysis(config: Dict):
    try:
        file = open(f"../results/{conf['run_name']}.pickle", 'rb')
    except FileNotFoundError:
        return
    repertoire = pickle.load(file)

    best_idx = jnp.argmax(repertoire.fitnesses)
    best_genome = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
    actions_tracking, reward = analyze_policy_behavior(best_genome, config)
    actions_tracking_df = pd.DataFrame.from_records(actions_tracking)
    actions_tracking_df.columns = [f"action_{i}" for i in range(len(actions_tracking_df.columns))]
    actions_tracking_df["cumulative_reward"] = reward
    target_file = conf["run_name"] + "_actions.csv"
    actions_tracking_df.to_csv("../results/" + target_file, index=True)


if __name__ == '__main__':
    conf = {
        "solver": {
            "n_nodes": 50,
        },
        "seed": 0,
        "problem": "walker2d",
    }

    tasks = ["reacher", "swimmer", "hopper", "walker2d", "halfcheetah"]

    args = sys.argv[1:]
    for arg in args:
        k, value = arg.split("=")
        if k == "problem_id":
            conf["problem"] = tasks[int(value)]

    for pb in tasks:
        for w in [True, False]:
            for seed in range(10):
                conf["problem"] = pb
                conf["seed"] = seed
                conf["solver"]["w_fn"] = w
                conf["solver"]["w_in"] = not w
                w_txt = "w_fn" if w else "w_in"
                conf["run_name"] = f"CGP_wann_{w_txt}_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
                print(conf["run_name"])
                run_analysis(conf)
