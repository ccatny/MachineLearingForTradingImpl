""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import random as rand  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 


class QLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    This is a Q learner object.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			     			  	 
    :type num_states: int  		  	   		  		 			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			     			  	 
    :type num_actions: int  		  	   		  		 			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type alpha: float  		  	   		  		 			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type gamma: float  		  	   		  		 			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type rar: float  		  	   		  		 			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			     			  	 
    :type radr: float  		  	   		  		 			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			     			  	 
    :type dyna: int  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """
    Q_table = np.zeros((100, 4))
    T_table = np.array(4) #state, action, state prime, reward
    state = 0
    action = 1
    state_prime = 2
    reward = 3

    def __init__(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        num_states=100,  		  	   		  		 			  		 			     			  	 
        num_actions=4,  		  	   		  		 			  		 			     			  	 
        alpha=0.2,  		  	   		  		 			  		 			     			  	 
        gamma=0.9,  		  	   		  		 			  		 			     			  	 
        rar=0.5,  		  	   		  		 			  		 			     			  	 
        radr=0.99,  		  	   		  		 			  		 			     			  	 
        dyna=0,  		  	   		  		 			  		 			     			  	 
        verbose=False,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
        self.num_actions = num_actions  		  	   		  		 			  		 			     			  	 
        self.s = 0  		  	   		  		 			  		 			     			  	 
        self.a = 0
        self.num_states = num_states
        self.rar = rar
        self.alpha = alpha
        self.Q_table = np.zeros([num_states, num_actions])
        self.T_table = np.zeros([num_states, num_actions, num_states])
        self.R_table = np.zeros([num_states, num_actions])
        self.dyna = dyna
        self.gamma = gamma
        self.radr = radr
        self.record = []
        self.state = 0
        self.action = 1
        self.state_prime = 2
        self.reward = 3

    def author(self):
        return 'czhang669'

    def selectAction(self, s):
        actionNum = 0
        if np.random.random() < self.rar:
            actionNum = np.random.randint(0, self.num_actions)
        else:
            actionNum = np.argmax(self.Q_table[s])
        return actionNum

    def querysetstate(self, s):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param s: The new state  		  	   		  		 			  		 			     			  	 
        :type s: int  		  	   		  		 			  		 			     			  	 
        :return: The selected action  		  	   		  		 			  		 			     			  	 
        :rtype: int  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.s = s  		  	   		  		 			  		 			     			  	 
        action = self.selectAction(s)
        self.a = action
        if self.verbose:  		  	   		  		 			  		 			     			  	 
            print(f"s = {s}, a = {action}")  		  	   		  		 			  		 			     			  	 
        return action

    def update_Q_Table(self, s, a, s_prime, r):
        self.Q_table[s,a] = (1-self.alpha) * self.Q_table[s,a] + self.alpha * (r + self.gamma * np.max(self.Q_table[s_prime]))
  		  	   		  		 			  		 			     			  	 
    def query(self, s_prime, r):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Update the Q table and return an action  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param s_prime: The new state  		  	   		  		 			  		 			     			  	 
        :type s_prime: int  		  	   		  		 			  		 			     			  	 
        :param r: The immediate reward  		  	   		  		 			  		 			     			  	 
        :type r: float  		  	   		  		 			  		 			     			  	 
        :return: The selected action  		  	   		  		 			  		 			     			  	 
        :rtype: int  		  	   		  		 			  		 			     			  	 
        """
        #self.T_table[self.s, self.a, s_prime] = self.T_table[self.s, self.a, s_prime] + 1
        self.R_table[self.s, self.a] = (1-self.alpha)*self.R_table[self.s, self.a] + self.alpha*r
        self.update_Q_Table(self.s, self.a, s_prime, r)
        self.record.append([self.s, self.a, s_prime, r]) #T is updated implicitly here
        self.a = self.selectAction(s_prime)
        self.s = s_prime
        self.rar = self.rar * self.radr

        #action = rand.randint(0, self.num_actions - 1)

        for i in range(self.dyna):
            seed = np.random.randint(low=0, high=len(self.record))
            random_s, random_a, temp_s_prime, temp_r = self.record[seed]
            self.update_Q_Table(random_s, random_a,temp_s_prime, temp_r)
        action = self.a
        if self.verbose:  		  	   		  		 			  		 			     			  	 
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
