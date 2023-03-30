""""""
"""Assess a betting strategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Chengkai Zhang (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: czhang669 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903614675 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""

import numpy as np
import matplotlib.pyplot as plt
import os

upperLimit = 80
episodeSize = 1000
prob = 18.0 / 38.0
limit = -256

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "czhang669"  # replace tb34 with your Georgia Tech username.


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903614675  # replace with your GT ID number


def get_spin_result(win_prob):
    """
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.

    :param win_prob: The probability of winning
    :type win_prob: float
    :return: The result of the spin.
    :rtype: bool
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def runAnEpisode(episodeAmount, prob):
    betRecord = np.zeros(episodeAmount + 1)
    balanceRecord = np.zeros(episodeAmount + 1)
    episodeWinnings = 0
    betCount = 0
    while episodeWinnings < upperLimit and betCount < episodeAmount :
        won = False
        betAmount = 1
        while won == False and betCount < episodeAmount :
            betCount = betCount + 1
            betRecord[betCount] = betAmount
            won = get_spin_result(prob)
            if won == True:
                episodeWinnings = episodeWinnings + betAmount
            else:
                episodeWinnings = episodeWinnings - betAmount
                betAmount = betAmount * 2
            balanceRecord[betCount] = episodeWinnings
    if betCount < episodeAmount:
        balanceRecord[betCount:] = upperLimit
    return betRecord, balanceRecord

def runAnEpisodeWithLimit(episodeAmount, prob, limit = -256):
    betRecord = np.zeros(episodeAmount + 1)
    balanceRecord = np.zeros(episodeAmount + 1)
    episodeWinnings = 0
    betCount = 0
    while episodeWinnings < upperLimit and betCount < episodeAmount and episodeWinnings > limit:
        won = False
        betAmount = 1
        while won == False and episodeWinnings > limit and betCount < episodeAmount :
            betCount = betCount + 1
            betRecord[betCount] = betAmount
            won = get_spin_result(prob)
            if won == True:
                episodeWinnings = episodeWinnings + betAmount
            else:
                episodeWinnings = max(episodeWinnings - betAmount, limit)
                betAmount = min(betAmount * 2, episodeWinnings - limit)
            balanceRecord[betCount] = episodeWinnings
    if betCount < episodeAmount:
        if episodeWinnings == limit:
            balanceRecord[betCount:] = limit
        else:
            balanceRecord[betCount:] = upperLimit
    return betRecord, balanceRecord

def exper1(size = 10):
    fig, ax = plt.subplots()
    ax.set(xlim=[0, 300]
           , ylim=[-256, 100]
           , title="Experiment 1.1, the records of winning of 10 episodes"
           , xlabel='Spins'
           , ylabel='Winnings')

    for i in range(size):
        betRecord, balanceRecord = runAnEpisode(episodeSize, prob)
        ax.plot(balanceRecord)
    plt.legend(["Episode " + str(i) for i in range(1, 11)])
    plt.savefig("figure_1.png")

def exper2and3 (size = 1000):
    records = np.zeros((size, episodeSize + 1), dtype=int)
    for i in range (size):
        betRecord, balanceRecord = runAnEpisode(episodeSize, prob)
        records[i] = balanceRecord
    mean = np.mean(records, axis=0)
    dev = np.std(records, axis=0)
    med = np.median(records, axis=0)
    #np.savetxt("records.txt", records, fmt='%f', delimiter=',')

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 300]
           , ylim=[-256, 100]
           , title="Experiment 1.2, the mean and stddev of winning of 1000 episodes"
           , xlabel='Spins'
           , ylabel='Winnings')
    ax.plot(mean + dev)
    ax.plot(mean)
    ax.plot(mean - dev)
    plt.legend(["Mean plus stddev", "Mean", "Mean minus stddev"])
    plt.savefig('figure_2.png')

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 300]
           , ylim=[-256, 100]
           , title="Experiment 1.3, the median and stddev of winning of 1000 episodes"
           , xlabel='Spins'
           , ylabel='Winnings')

    ax.plot(med + dev)
    ax.plot(med)
    ax.plot(med - dev)
    plt.legend(["median plus stddev", "median", "median minus stddev"])
    plt.savefig('figure_3.png')

def exper4and5(size = 1000):
    records = np.zeros((size, episodeSize + 1), dtype=int)
    for i in range(size):
        betRecord, balanceRecord = runAnEpisodeWithLimit(episodeSize, prob, limit)
        records[i] = balanceRecord
    mean = np.mean(records, axis=0)
    dev = np.std(records, axis=0)
    med = np.median(records, axis=0)

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 300]
           , ylim=[-256, 100]
           , title="Experiment 2.1, the mean and stddev of winning of 1000 episodes"
           , xlabel='Spins'
           , ylabel='Winnings')

    ax.plot(mean + dev)
    ax.plot(mean)
    ax.plot(mean - dev)
    plt.legend(["Mean plus stddev", "Mean", "Mean minus stddev"])
    plt.savefig('figure_4.png')

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 300]
           , ylim=[-256, 100]
           , title="Experiment 2.2, the median and stddev of winning of 1000 episodes"
           , xlabel='Spins'
           , ylabel='Winnings')

    ax.plot(med + dev)
    ax.plot(med)
    ax.plot(med - dev)
    plt.legend(["Median plus stddev", "Median", "Median minus stddev"])
    plt.savefig('figure_5.png')


def test_code():
    """
    Method to test your code
    """
    win_prob = 0.60  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # add your code here to implement the experiments
    exper1(10)
    exper2and3(1000)
    exper4and5(1000)


if __name__ == "__main__":
    test_code()