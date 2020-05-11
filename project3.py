import csv
import numpy as np
import pandas as pd


#Define the game
def Generalized_Second_Price_Auction(n, values, bids, prob):
    payoffs = []
    return payoffs

#payoffs: matrix of all payoff, epsilon: learning_rate, h: max payoff
def exponential_weight_full_info(payoffs, epsilon, h):
    n=payoffs.shape[1]
    k=payoffs.shape[0]
    action = []
    sum = 0
    action.append(np.random.randint(0,k))
    sum =+ payoffs[action[0]][0]
    v_payoff_sum = payoffs[:,0]
    for j in range(1,n):
        exp = []
        for i in range(0,k):
            prob = (1+epsilon)**(v_payoff_sum[i]/h)
            exp.append(prob)
        exp_sum = np.sum(np.array(exp))
        exp_prob = np.array(exp)/exp_sum
        v_payoff_sum = v_payoff_sum + payoffs[:,i]
        #print(exp_prob)
        rand = np.random.rand()
        select_action = 0
        cdf = exp_prob[0]
        while(rand > cdf and select_action < k-1):
            select_action += 1
            cdf = cdf + exp_prob[select_action]
        action.append(select_action)
        sum += payoffs[select_action][j]
    return action, sum


def follow_the_perturbed_leader_full_info(payoffs, epsilon, h):
    n=payoffs.shape[1]
    k=payoffs.shape[0]
    action = []
    sum = 0
    perturbation_value_array = np.zeros(k)
    for i in range(0,k):
        perturbation_value_array[i] = h * np.random.exponential(epsilon)
    #print(perturbation_value_array)
    for j in range(1,n):
        action_hist_payoff = np.zeros(k)
        for i in range(0,k):
            action_hist_payoff[i] = np.sum(payoffs[i][0:j])+ perturbation_value_array[i]
        selected_action = np.argmax(action_hist_payoff)
        action.append(selected_action)
        sum += payoffs[selected_action][j]
    return action, sum









