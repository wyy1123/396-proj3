import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from project3 import Generalized_Second_Price_Auction, exponential_weight_historical_info, follow_the_perturbed_leader_historical_info

#creating payoff matrix:
bid_space = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
payoff_table = []
algo = "EW"

def choose_action(algo,payoffs,epsilon,h):
    cur_action_idx = 0
    actions =[]
    total_payoff = 0
    n=payoffs.shape[1]
    k=payoffs.shape[0]
    total_payoff = 0
    if algo =="EW":
        cur_action_idx,actions= exponential_weight_full_info(payoffs, epsilon, h)
    if algo =="FTPL":
        cur_action_idx,actions= follow_the_perturbed_leader_full_info(payoffs, epsilon, h)
        
    return cur_action_idx

class player():
    def __init__(self,algo,epsilon,h,action_space):
        self.algo = algo
        self.epsilon = epsilon
        self.value = h
        self.action_space = action_space
        self.payoffs = []
        self.actions =[]   # Keep track of past action
        if algo =="EW":
            for i in range(0,len(action_space)):
                self.payoffs.append([0])
            #?: for ew is it initialized to be 0?
        if algo =="FTPL":
            for i in range(0,len(action_space)):
                self.payoffs.append([h * np.random.exponential(epsilon)])
            #?: what are the initial values for ftpl?
               
    def generate_round_payoff(self,bid, bids):
        round_payoff = np.zeros(len(self.action_space))      # changed from "round_payoff = np.zeros(20)" 
        for i in range(len(self.action_space)):                      # changed from "len(bid_space)"
            if self.action_space[i] < np.amax(np.asarray(bids)):
                round_payoff[i] = 0
            if self.action_space[i] == np.amax(np.asarray(bids)):
                round_payoff[i] = 0.5*(self.value-bid)
            if self.action_space[i] > np.amax(np.asarray(bids)):
                round_payoff[i] = self.value-bid  
        return round_payoff

    def update_payoff(self,round_payoff):
        for i in range(len(self.payoffs)):
            self.payoffs[i].append(round_payoff[i])       
    
    def choose_action(self):
        cur_action_idx = 0
        #actions =[]
        n=len(self.payoffs[0])
        k=len(self.payoffs)
        total_payoff = 0
        if algo =="EW":
            cur_action_idx = exponential_weight_historical_info(np.asarray(self.payoffs), self.epsilon, self.value, self.actions)
        if algo =="FTPL":
            cur_action_idx,total_payoff= follow_the_perturbed_leader_historical_info(np.asarray(self.payoffs), self.epsilon, self.value, self.actions)
        self.actions.append(cur_action_idx)  #Keep track of each actions
        return cur_action_idx

#1. initialize the player classes
#for each player:
#  2. in each round, choose action
#  then record total payoff 
#  update the per round payoff based on all players' bids
#  

#initializing the player here:
"""
EW_player1 = player("EW",0.5,20,bid_space)
EW_player2 = player("EW",0.5,40,bid_space)
EW_player3 = player("EW",0.5,60,bid_space)
EW_player4 = player("EW",0.5,80,bid_space)
EW_player5 = player("EW",0.5,100,bid_space)
FTPL_player1 = player("FTPL",0.5,20,bid_space)
FTPL_player2 = player("FTPL",0.5,40,bid_space)
FTPL_player3 = player("FTPL",0.5,60,bid_space)
FTPL_player4 = player("FTPL",0.5,80,bid_space)
FTPL_player5 = player("FTPL",0.5,100,bid_space)
"""

#linear test
"""
EW_player1 = player("EW",5,20,[0,5,10,15,20])
EW_player2 = player("EW",5,40,[0,5,10,15,20,25,30,35,40])
EW_player3 = player("EW",5,60,[0,5,10,15,20,25,30,35,40,45,50,55,60])
EW_player4 = player("EW",5,80,[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80])
EW_player5 = player("EW",5,100,[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
"""

#Geometric test

print(math.log(20, 1.5))
print(math.log(40, 1.5))
print(math.log(60, 1.5))
print(math.log(80, 1.5))
print(math.log(100, 1.5))
EW_player1 = player("EW",5,20,[1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7])
EW_player2 = player("EW",5,40,[1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7, 1.5**8,1.5**9])
EW_player3 = player("EW",5,60,[1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7, 1.5**8,1.5**9, 1.5**10])
EW_player4 = player("EW",5,80,[1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7, 1.5**8,1.5**9, 1.5**10])
EW_player5 = player("EW",5,100,[1.5, 1.5**2, 1.5**3, 1.5**4, 1.5**5, 1.5**6, 1.5**7, 1.5**8,1.5**9, 1.5**10, 1.5**11])






n = 2000
bid = bid_space
players = [EW_player1, EW_player2, EW_player3, EW_player4, EW_player5]

for i in range(0,n):
    
    bids = []

    for p in players:
        algo_bid = p.action_space[p.choose_action()]
        bids.append(algo_bid)

    print(bids)
    for p in players:
        round_payoffs = p.generate_round_payoff(p.action_space[p.actions[-1]], bids)
        p.update_payoff(round_payoffs)



    







