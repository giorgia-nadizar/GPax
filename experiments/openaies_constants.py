import pickle
import sys
import time
from functools import partial
from typing import Dict
from typing import Tuple

import jax
import jax.numpy as jnp
import qdax.tasks.brax.v1 as environments
from evosax import Strategies
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
from jax.flatten_util import ravel_pytree
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Genotype
from qdax.tasks.brax.v1.env_creators import get_mask_from_transitions
from qdax.utils.metrics import CSVLogger

from gpax.gp.cartesian_genetic_programming import CGP


def reopt_single_genome_openai_es(genome: Genotype, config: Dict) -> Tuple[Genotype, jnp.ndarray]:
    env_name = config["problem"]

    num_iterations = 100
    population_size = 128

    key = jax.random.key(config["seed"])

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
        es_config={"maximize": False,
                   "centered_rank": True,
                   "lrate_init": 0.01,
                   "lrate_decay": 0.999,
                   "lrate_limit": 0.001},
        es_params={"sigma_init": 0.05,
                   "sigma_decay": 0.999,
                   "sigma_limit": 0.01},
        seed=0,
    )

    reset_key, key = jax.random.split(key)
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

        # Evaluate
        fitnesses = jnp.sum(transitions.rewards * (1.0 - mask), axis=1)

        return -fitnesses

    for _ in range(num_iterations):
        # sample
        key, subkey = jax.random.split(key)
        samples = solver.ask()
        scores = fitness_fn(samples)
        solver.tell(fitness=scores)

    # max_id = jnp.argmax(scores)
    max_id = jnp.argmin(scores)
    best_score = scores[max_id]
    best_params_array = samples[max_id]
    best_params_pytree = weights_tree_def(best_params_array)
    return policy_graph_structure.update_weights(genome, best_params_pytree), best_score


def run_openai_es_constants_reopt(config: Dict):
    try:
        file = open(f"../results/{conf['run_name']}.pickle", 'rb')
    except FileNotFoundError:
        return
    repertoire = pickle.load(file)
    reopt_fn = partial(reopt_single_genome_openai_es, config=config)

    updated_genomes = []
    updated_fitnesses = []

    start_time = time.time()
    for idx in range(len(repertoire.fitnesses)):
        genome = jax.tree.map(lambda x: x[idx], repertoire.genotypes)
        updated_genome, new_fitness = reopt_fn(genome)
        updated_genomes.append(updated_genome)
        updated_fitnesses.append(new_fitness)
        print(idx, new_fitness)
    timelapse = time.time() - start_time

    stacked_genomes = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *updated_genomes)
    fitnesses_array = jnp.asarray(updated_fitnesses)

    repertoire_to_store = GARepertoire(
        genotypes=stacked_genomes,
        fitnesses=fitnesses_array,
        extra_scores={},
        keys_extra_scores=None
    )

    target_file = conf["run_name"].replace("CGP_wann", f"CGP_wann-reopt3")
    path = f"../results/{target_file}.pickle"
    with open(path, 'wb') as file:
        pickle.dump(repertoire_to_store, file)

    csv_logger = CSVLogger(
        f'../results/{target_file}.csv',
        header=["iteration", "max_fitness", "time"]
    )
    metrics = {
        "iteration": 0,
        "max_fitness": jnp.max(fitnesses_array),
        "time": timelapse
    }

    # Log metrics
    csv_logger.log(metrics)


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

    for w in [True, False]:
        for seed in range(10):
            conf["seed"] = seed
            conf["solver"]["w_fn"] = w
            conf["solver"]["w_in"] = not w
            w_txt = "w_fn" if w else "w_in"
            conf["run_name"] = f"CGP_wann_{w_txt}_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            print(conf["run_name"])
            run_openai_es_constants_reopt(conf)
