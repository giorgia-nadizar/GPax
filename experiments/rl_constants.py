import pickle
import sys
from functools import partial
from typing import Any, Tuple
from typing import Dict

import jax
import jax.numpy as jnp
import qdax.tasks.brax.v1 as environments
from brax.envs import State as EnvState
from qdax.baselines.td3 import TD3TrainingState
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer
from qdax.custom_types import Genotype

from gpax.gp.cartesian_genetic_programming import CGP
from gpax.rl.graph_td3 import GraphTD3Config, GraphTD3


def reopt_single_genome(genome: Genotype, config: Dict) -> Tuple[Genotype, jnp.ndarray]:
    env_batch_size = 16
    num_steps = 160_000
    warmup_steps = 10_000
    buffer_size = 100_000

    episode_length = 1000
    grad_updates_per_step = 1
    soft_tau_update = 0.005
    expl_noise = 0.1
    batch_size = 256
    policy_delay = 2
    discount = 0.99
    noise_clip = 0.5
    policy_noise = 0.2
    reward_scaling = 0.01 #1.0
    critic_hidden_layer_size = (256, 256)
    policy_hidden_layer_size = (256, 256)
    critic_learning_rate = 3e-4 / 4
    policy_learning_rate = 3e-4

    env_name = config["problem"]

    # Create training environment
    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
    )
    # Create eval environment
    eval_env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
        eval_metrics=True,
    )

    key = jax.random.key(config["seed"])
    key, subkey_1, subkey_2 = jax.random.split(key, 3)
    env_state = jax.jit(env.reset)(rng=subkey_1)
    eval_env_first_state = jax.jit(eval_env.reset)(rng=subkey_2)

    # Initialize buffer
    dummy_transition = Transition.init_dummy(
        observation_dim=env.observation_size, action_dim=env.action_size
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    # Init the CGP policy graph with default values
    policy_graph_structure = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        weighted_functions=True,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=jnp.tanh
    )

    graph_td3_config = GraphTD3Config(
        episode_length=episode_length,
        batch_size=batch_size,
        policy_delay=policy_delay,
        soft_tau_update=soft_tau_update,
        expl_noise=expl_noise,
        critic_hidden_layer_size=critic_hidden_layer_size,
        policy_hidden_layer_size=policy_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        policy_learning_rate=policy_learning_rate,
        policy_graph_structure=policy_graph_structure,
        policy_graph_genome=genome,
        discount=discount,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        reward_scaling=reward_scaling,
    )

    # Initialize TD3 algorithm
    td3 = GraphTD3(config=graph_td3_config, action_size=env.action_size)

    key, subkey = jax.random.split(key)
    training_state = td3.init(
        subkey, action_size=env.action_size, observation_size=env.observation_size
    )
    # training_state = TD3TrainingState(
    #     policy_optimizer_state=training_state.policy_optimizer_state,
    #     policy_params=policy_graph_structure.get_weights(genome),
    #     critic_optimizer_state=training_state.critic_optimizer_state,
    #     critic_params=training_state.critic_params,
    #     target_policy_params=training_state.target_policy_params,
    #     target_critic_params=training_state.target_critic_params,
    #     key=training_state.key,
    #     steps=training_state.steps,
    # )

    # Wrap and jit play step function
    play_step = partial(
        td3.play_step_fn,
        env=env,
        deterministic=False,
    )

    # Wrap and jit play eval step function
    play_eval_step = partial(td3.play_step_fn, env=eval_env, deterministic=True)

    # Wrap and jit eval policy function
    eval_policy = partial(
        td3.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
    )

    # Wrap and jit do iteration function
    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=td3.update,
    )

    def _scan_do_iteration(
            carry: Tuple[TD3TrainingState, EnvState, ReplayBuffer],
            unused_arg: Any,
    ) -> Tuple[Tuple[TD3TrainingState, EnvState, ReplayBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Evaluate untrained policy
    true_return, true_returns = eval_policy(training_state=training_state)
    print(true_return)

    # Warmstart the buffer
    replay_buffer, env_state, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
        env_state=env_state,
        num_warmstart_steps=warmup_steps,
        env_batch_size=env_batch_size,
        play_step_fn=play_step,
    )

    total_num_iterations = num_steps // env_batch_size
    # print(training_state)

    # Main training loop: update agent, evaluate and log metrics
    scan_iterations = total_num_iterations // 500
    for x in range(scan_iterations):
        (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
            _scan_do_iteration,
            (training_state, env_state, replay_buffer),
            (),
            length=500,
        )

        # Evaluate
        final_true_return, final_true_returns = eval_policy(training_state=training_state)
        better_params = training_state.policy_params
        print(x, final_true_return)
        print(metrics)

    return true_return, policy_graph_structure.update_weights(genome, better_params), final_true_return


def run_rl_constants_reopt(config: Dict):
    try:
        file = open(f"../results/{conf['run_name']}.pickle", 'rb')
    except FileNotFoundError:
        return
    repertoire = pickle.load(file)
    reopt_fn = partial(reopt_single_genome, config=config)
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_genome = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
    print(repertoire.fitnesses[best_idx])
    # print(best_genome)
    # result = jax.vmap(reopt_fn)(repertoire.genotypes)
    reopt_fn(best_genome)


if __name__ == '__main__':
    conf = {
        "solver": {
            "n_nodes": 50,
        },
        "seed": 0,
        "problem": "hopper",
    }

    tasks = ["reacher", "swimmer", "hopper", "walker2d", "halfcheetah"]

    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split("=")
        if key == "problem_id":
            conf["problem"] = tasks[int(value)]

    for seed in range(1):
        conf["seed"] = seed
        conf["run_name"] = "CGP_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
        print(conf["run_name"])
        run_rl_constants_reopt(conf)
