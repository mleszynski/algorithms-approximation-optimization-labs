# opengym.py
"""Volume 2: Open Gym
Marcelo Leszynski
Math 323 Sec 003
02/20/21
"""

import gym
import numpy as np
from IPython.display import clear_output
import random
import time
import math

def find_qvalues(env,alpha=.1,gamma=.6,epsilon=.1):
    """
    Use the Q-learning algorithm to find qvalues.

    Parameters:
        env (str): environment name
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): maximum value

    Returns:
        q_table (ndarray nxm)
    """
    # Make environment
    env = gym.make(env)
    # Make Q-table
    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    # Train
    for i in range(1,100001):
        # Reset state
        state = env.reset()

        epochs, penalties, reward, = 0,0,0
        done = False

        while not done:
            # Accept based on alpha
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, info = env.step(action)

            # Calculate new qvalue
            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Check if penalty is made
            if reward == -10:
                penalties += 1

            # Get next observation
            state = next_state
            epochs += 1

        # Print episode number
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.")
    return q_table

# Problem 1
def random_blackjack(n):
    """
    Play a random game of Blackjack. Determine the
    percentage the player wins out of n times.

    Parameters:
        n (int): number of iterations

    Returns:
        percent (float): percentage that the player
                         wins
    """
    # initialize environment ###################################################
    env = gym.make('Blackjack-v0')
    num_wins = 0

    # play n games #############################################################
    for i in range(n):
        env.reset()
        is_done = False
        while not is_done:
            observation = env.step(env.action_space.sample())
            # win condition ####################################################
            if observation[1] == 1:
                num_wins += 1
                break

            is_done = observation[2]

    # return win percentage ####################################################
    env.close()
    return num_wins / n


# Problem 2
def blackjack(n=11):
    """
    Play blackjack with naive algorithm.

    Parameters:
        n (int): maximum accepted player hand

    Return:
        percent (float): percentage of 10000 iterations
                         that the player wins
    """
    # initialize environment ###################################################
    env = gym.make('Blackjack-v0')
    num_wins = 0

    # play 10000 games #########################################################
    for i in range(10000):
        # start environment and draw first card ################################
        env.reset()
        observation = env.step(0)

        # play game until end ##################################################
        while True:
            # check for naive algorithm condition ##############################
            if observation[0][0] <= n:
                observation = env.step(1)
            else:
                observation = env.step(0)
            
            if observation[2] == True:
                if observation[1] == 1:
                    num_wins += 1
                break

    # return win percentage ####################################################
    env.close()
    return num_wins / 10000
    # optimal n is around 15


# Problem 3
def cartpole():
    """
    Solve CartPole-v0 by checking the velocity
    of the tip of the pole

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    # initialize environment ###################################################
    env = gym.make('CartPole-v0')

    try:
        env.reset()
        num_steps = 1

        # start the simulation #################################################
        observation = env.step(env.action_space.sample())

        while not observation[2]:
            if observation[0][3] >= 0:
                observation = env.step(1)
            else:
                observation = env.step(0)
            num_steps += 1
            env.render()

    finally:
        env.close()

    return num_steps


# Problem 4
def car():
    """
    Solve MountainCar-v0 by checking the position
    of the car.

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    # initialize environment ###################################################
    env = gym.make('MountainCar-v0')
    num_steps = 1

    try:
        env.reset()

        # run the simulation ###################################################
        observation = env.step(0)

        while not observation[2]:
            if observation[0][1] > 0:
                moving_right = True
            else:
                moving_right = False

            if moving_right:
                observation = env.step(2)
            else:
                observation = env.step(0)

            num_steps += 1
            env.render()

    finally:
        env.close()

    return num_steps 


# Problem 5
def taxi(q_table):
    """
    Compare naive and q-learning algorithms.

    Parameters:
        q_table (ndarray nxm): table of qvalues

    Returns:
        naive (float): mean reward of naive algorithm
                       of 10000 runs
        q_reward (float): mean reward of Q-learning algorithm
                          of 10000 runs
    """
    # initialize environment ###################################################
    env = gym.make('Taxi-v3')
    num_experiments = 10000
    rand_rewards = 0
    q_rewards = 0

    try:
        # simulate random movement 10000 times #################################
        for i in range(num_experiments):
            env.reset()
            while True:
                # run taxi until simulation is over, then append score #########
                observation = env.step(env.action_space.sample())
                rand_rewards += observation[1]
                if observation[2]:
                    break

        # simulate smart movement 10000 times ##################################
        for i in range(num_experiments):
            state = env.reset()

            while True:
                # run taxi until end of q_table is reached, then append score ######
                observation = env.step(np.argmax(q_table[state]))
                state = observation[0]

                q_rewards += observation[1]
                if observation[2]:
                    break
    finally:
        env.close()

    return rand_rewards / num_experiments, q_rewards / num_experiments
