"""Learning script for multi-agent problems.
Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:

    http://127.0.0.1:8265

"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.append('../')
import numpy as np
import torch
import ray
from ray import tune
from ray.tune import register_env, CLIReporter
from ray.rllib.agents import ppo
from utils import build_env_by_name
from experiments.learning import shared_constants
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import str2bool, sync

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones', default=8, type=int, help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--obs', default='kin', type=ObservationType, help='Observation space (default: kin)',
                        metavar='')
    parser.add_argument('--act', default='vel', type=ActionType, help='Action space (default: vel)',
                        metavar='')
    parser.add_argument('--algo', default='cc', type=str, choices=['cc'], help='MARL approach (default: cc)',
                        metavar='')
    parser.add_argument('--workers', default=1, type=int, help='Number of RLlib workers (default: 1)', metavar='')
    parser.add_argument('--debug', default=False, type=str2bool,
                        help='Run in one Thread if true, for debugger to work properly', metavar='')
    parser.add_argument('--gui', default=False, type=str2bool,
                        help='Enable gui rendering', metavar='')
    parser.add_argument('--exp', type=str,
                        help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>',
                        metavar='')

    ARGS = parser.parse_args()

    #### Save directory ########################################

    #### Print out current git commit hash #####################
    # if platform == "linux" or platform == "darwin":
    #    git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #    with open(filename + '/git_commit.txt', 'w+') as f:
    #        f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif ARGS.obs == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit(2)
    else:
        print("[ERROR] unknown ObservationType")
        exit(3)
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ARGS.act == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit(4)

    if ARGS.num_drones % 2 == 0 and ARGS.num_drones >= 2:

        INIT_XYZS = np.array([[0.0,0.0,0.0]]*ARGS.num_drones)
        for i in range(ARGS.num_drones):
            if i%2 == 0:
                INIT_XYZS[i] = [0.5 * (i/2), 5, 2]
            else:
                INIT_XYZS[i] = [0.5 * ((i-1)/2), -5, 2]

        INIT_RPYS = np.array([[0.0,0.0,0.0]]*ARGS.num_drones)
        for i in range(ARGS.num_drones):
            if i%2 == 0:
                INIT_RPYS[i] = [0, 0, -1.41372]
            else:
                INIT_RPYS[i] = [0, 0, 1.41372]

    else:
        logging.exception("The number of drones must be even and grater than2")
        exit(-1)

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True, local_mode=ARGS.debug)
    env = "BattleAviary"

    import importlib

    # use only env, using module cause errors in train
    module = importlib.import_module(env)
    # module = importlib.import_module('exercise-2.' + env)
    env_class_imported = getattr(module, env)

    env_callable, obs_space, act_space, temp_env = build_env_by_name(env_class=env_class_imported,
                                                                     exp=ARGS.exp,
                                                                     num_drones=ARGS.num_drones,
                                                                     aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                     obs=ARGS.obs,
                                                                     initial_xyzs=INIT_XYZS,
                                                                     initial_rpys=INIT_RPYS,
                                                                     act=ARGS.act,
                                                                     gui=ARGS.gui
                                                                     )
    #### Register the environment ##############################
    register_env(env, env_callable)

    config = {
        "env": env,
        # "no_done_at_end": True,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": torch.cuda.device_count(),
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "entropy_coeff": tune.loguniform(0.00000001, 0.1),
        "lr": tune.loguniform (5e-5, 1),
        "sgd_minibatch_size": tune.choice ([ 32, 64, 128, 256, 512]),
        "lambda": tune.choice ([0.1,0.3, 0.5, 0.7,0.9,1.0]),
        "multiagent": {
            # We only have one policy (calling it "shared").
            # Class, obs/act-spaces, and config will be derived
            # automatically.
            "policies": {
                "pol0": (None, obs_space[0], act_space[0], {"agent_id": 0, }),
                "pol1": (None, obs_space[1], act_space[1], {"agent_id": 1, }),
            },
            "policy_mapping_fn": lambda x: "pol" + str(x%2),
            # Always use "shared" policy.
        }
    }

    stop = {
        "timesteps_total": 20000,  # 100000 ~= 10'
        # "episode_reward_mean": 0,
        # "training_iteration": 100,
    }

    if not ARGS.exp:

        filename = os.path.dirname(os.path.abspath(__file__)) + '/results/save' + '-' + str(
            ARGS.num_drones) + '-' + ARGS.algo + '-' + ARGS.obs.value + '-' + ARGS.act.value + '-' + datetime.now().strftime(
            "%m.%d.%Y_%H.%M.%S")
        if not os.path.exists(filename):
            os.makedirs(filename + '/')
        # Using trainer.train the dubugger works better
        if ARGS.debug:

            agent = ppo.PPOTrainer(config=config)
            while True:
                print(agent.train())
        else:
            results = tune.run(
                "PPO",
                stop=stop,
                config=config,
                verbose=True,
                progress_reporter=CLIReporter(max_progress_rows=10),
                # checkpoint_freq=50000,
                checkpoint_at_end=True,
                local_dir=filename,
            )

            # check_learning_achieved(results, 1.0)

            #### Save agent ############################################
            checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                           mode='max'
                                                                                           ),
                                                              metric='episode_reward_mean'
                                                              )
            with open(filename + '/checkpoint.txt', 'w+') as f:
                f.write(checkpoints[0][0])

            # print(checkpoints)

    else:
        
        if (ARGS.exp.split("-")[4] == 'kin'):
            OBS = ObservationType.KIN 
        else:  
            OBS = ObservationType.RGB

        action_name = ARGS.exp.split("-")[5]
        NUM_DRONES = int(ARGS.exp.split("-")[2])
        ACT = [action for action in ActionType if action.value == action_name][0]
        #### Restore agent #########################################
        agent = ppo.PPOTrainer(config=config)
        with open(ARGS.exp + '/checkpoint.txt', 'r+') as f:
            checkpoint = f.read()
        agent.restore(checkpoint)
        print(checkpoint)

        #### Extract and print policies ############################
        policy0 = agent.get_policy("pol0")
        policy1 = agent.get_policy("pol1")
        
        #### Show, record a video, and log the model's performance #
        obs = temp_env.reset()
        logger = Logger(logging_freq_hz=int(temp_env.SIM_FREQ / temp_env.AGGR_PHY_STEPS),
                        num_drones=NUM_DRONES
                        )
        if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            action = {i: np.array([0]) for i in range(NUM_DRONES)}
        elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
        elif ACT == ActionType.PID:
            action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
        else:
            print("[ERROR] unknown ActionType")
            exit()
        start = time.time()
        for i in range(6 * int(temp_env.SIM_FREQ / temp_env.AGGR_PHY_STEPS)):  # Up to 6''
            #### Deploy the policies ###################################
            temp = {}
            temp[0] = policy0.compute_single_action(
                np.hstack(obs[0]))  # Counterintuitive order, check params.json
            temp[1] = policy1.compute_single_action(np.hstack(obs[1]))
            if NUM_DRONES == 8:
                temp[2] = policy0.compute_single_action(np.hstack(obs[2]))
                temp[3] = policy1.compute_single_action(np.hstack(obs[3]))
                temp[4] = policy0.compute_single_action(np.hstack(obs[4]))
                temp[5] = policy1.compute_single_action(np.hstack(obs[5]))
                temp[6] = policy0.compute_single_action(np.hstack(obs[6]))
                temp[7] = policy1.compute_single_action(np.hstack(obs[7]))

            if NUM_DRONES == 8:
                action = {0: temp[0][0], 1: temp[1][0], 2: temp[2][0], 3: temp[3][0], 4: temp[4][0], 5: temp[5][0], 6: temp[6][0], 7: temp[7][0]}
            elif NUM_DRONES == 2:
                action = {0: temp[0][0], 1: temp[1][0]}
            obs, reward, done, info = temp_env.step(action)
            temp_env.render()
            if OBS == ObservationType.KIN:
                for j in range(NUM_DRONES):
                    logger.log(drone=j,
                               timestamp=i / temp_env.SIM_FREQ,
                               state=np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(action[j], (4))]),
                               control=np.zeros(12)
                               )
            sync(np.floor(i * temp_env.AGGR_PHY_STEPS), start, temp_env.TIMESTEP)
            # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
        temp_env.close()
        logger.save_as_csv("ma")  # Optional CSV save
        logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
