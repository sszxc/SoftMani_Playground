#!/usr/bin/env python

import datetime
import os
import time
import argparse
import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ravens import Dataset, Environment, DualArmEnvironment, agents, tasks
from ravens import utils


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',           default=21)
    parser.add_argument('--gpu',            default='0')
    parser.add_argument('--task',           default='cable-vessel')
    args = parser.parse_args()

    # GPU setup (skiped)

    # Initialize Environment
    task = tasks.names[args.task]()
    env = DualArmEnvironment(disp=True, hz=240.0)
    utils.cprint('Finished env init. Starting rollout...', 'green')

    # Start one rollout
    np.random.seed(args.seed)
    obs = env.reset(task)
    # breakpoint()

    agent = task.oracle(env)

    info = env.info
    for t in range(task.max_steps):
        act = agent.act(obs, info)
        (obs, reward, done, info) = env.step(act)
        last_obs_info = (obs, info)
        #print(info['extras'], info['...']) # Use this to debug if needed.
        if done:
            break
    utils.cprint(f'Rollout complete after {t+1} steps.', 'green')
    print(F'Last obs: {last_obs_info[0]}')
