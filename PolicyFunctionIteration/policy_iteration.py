# policy_iteration.py
"""Volume 2: Policy Function Iteration.
Marcelo Leszynski
Math 323 Sec 003
04/06/21
"""

import gym
from gym import wrappers
import numpy as np

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]


# Problem 1
def value_iteration(P, nS, nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    V_new=np.zeros(nS)

    # find the next V values ###################################################
    for i in range(maxiter):  # iterate maxiter times or less
        V_old = np.copy(V_new)

        # iterate through possible states ######################################
        for s in range(nS):
            sa_vector = np.zeros(nA)

            # iterate through possible actions #################################
            for a in range(nA):
                for tuple_info in P[s][a]:
                    p, s_, u, cont = tuple_info
                    sa_vector[a] += (p * (u + beta * V_old[s_]))
            V_new[s] = np.max(sa_vector)

        # check for terminating conditions #####################################
        if np.linalg.norm(V_new - V_old) < tol:
            return V_new, i+1

    # non-convergent case ######################################################
    return V_new, maxiter


# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    # initialize policy vector #################################################
    policy = np.zeros(nS, dtype='int')

    # calculate value function argmax values ###################################
    # iterate through possible states ##########################################
    for s in range(len(policy)):
        sa_vector = np.zeros(nA)

        # iterate through possible actions #####################################
        for a in range(nA):
            for tuple_info in P[s][a]:
                p, s_, u, cont = tuple_info
                sa_vector[a] += (p * (u + beta * v[s_]))

        # calculate optimizing policy ##########################################
        policy[s] = int(np.argmax(sa_vector))

    return policy 


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    V_new = np.zeros(nS)

    # find the next V values ###################################################
    while True:
        V_old = np.copy(V_new)

        # iterate through possible states ######################################
        for s in range(nS):
            sa_list = list()
            # only have to check optimal actions ###############################
            for tuple_info in P[s][policy[s]]:
                p, s_, u, cont = tuple_info
                sa_list.append((p * (u + beta * V_old[s_])))
            V_new[s] = np.max(sa_list)

        if np.linalg.norm(V_new - V_old) < tol:
            return V_new


# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    # initialize policy and value vectors ######################################
    policy_new = np.zeros(nS, dtype='int')
    v = np.zeros(nA)

    # perform policy iteration algorithm #######################################
    for i in range(maxiter):
        policy_old = np.copy(policy_new)
        v = compute_policy_v(P, nS, nA, policy_new, beta, tol)
        policy_new = extract_policy(P, nS, nA, v, beta)

        # check for terminating case ###########################################
        if np.linalg.norm(policy_new-policy_old) < tol:
            return v, policy_new, i + 1
        
    return v, policy_new, maxiter


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    # initialize environment and beginning variables ###########################
    env_name='FrozenLake8x8-v0'
    if basic_case:
        env_name='FrozenLake-v0'

    env=gym.make(env_name).env
    nS=env.nS
    nA=env.nA
    P=env.P

    # calculate value and policy iteration vectors #############################
    vi_policy=extract_policy(P, nS, nA, value_iteration(P, nS, nA)[0])
    vi_total_rewards=0
    pi_value_func, pi_policy, n = policy_iteration(P, nS, nA)
    pi_total_rewards=0

    try:
    # run the value iteration policy M times and calculate total rewards #######
        for i in range(M):
            vi_total_rewards += run_simulation(env, vi_policy, render)

    # run the policy iteration policy M times and calculate total rewards ######
        for i in range(M):
            pi_total_rewards += run_simulation(env, pi_policy, render)

    # close environment and return desired values ##############################
    finally:
        env.close()

    return vi_policy, vi_total_rewards/M, pi_value_func, pi_policy, pi_total_rewards/M

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    # reset environment and initialize variables ###############################
    obs=env.reset()
    done=False
    num_iters=0

    # run the simulation until it is complete ##################################
    while not done:
        # optional: render the environment #####################################
        if render:
            env.render(mode='human')
        obs, reward, done, info = env.step(int(policy[obs]))
        num_iters += 1
        if done:
            return (beta**num_iters)*reward


#if __name__ == "__main__":
    # test prob 1 ##############################################################
    #print(value_iteration(P, 4, 4))
    ############################################################################


    # test prob 2 ##############################################################
    #print(extract_policy(P, 4, 4, value_iteration(P, 4, 4)[0]))
    ############################################################################


    # test prob 3 ##############################################################
    #print(compute_policy_v(P, 4, 4, extract_policy(P, 4, 4, value_iteration(P, 4, 4)[0])))
    ############################################################################


    # test prob 4 ##############################################################
    #print(policy_iteration(P, 4, 4))
    ############################################################################


    # test prob 5 ##############################################################
    #vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards = frozen_lake()
    #print('vi_policy =', vi_policy)
    #print('vi_total_rewards =', vi_total_rewards)
    #print('pi_value_func =\n', pi_value_func)
    #print('pi_policy =', pi_policy)
    #print('pi_total_rewards =', pi_total_rewards)
    ############################################################################


    # test prob 6 ##############################################################
    #vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards = frozen_lake(basic_case=True, render=True)
    #print('vi_policy =', vi_policy)
    #print('vi_total_rewards =', vi_total_rewards)
    #print('pi_value_func =\n', pi_value_func)
    #print('pi_policy =', pi_policy)
    #print('pi_total_rewards =', pi_total_rewards)
    ############################################################################