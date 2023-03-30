""""""

"""  		  	   		  		 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import LinRegLearner as lrl
import DTLearner as dl
import RTLearner as rl
import BagLearner as bl
import time as time
import sys


def get_data(file, trim_row_headers=True):
    """
    :param file: testing dataset
    :return: X and Y in same np array
    """
    with open(file) as f:
        alldata = np.genfromtxt(f, delimiter=",")
        # Cleaning
        if trim_row_headers:
            alldata = alldata[1:, 1:]  # drops row/date column and headers
        # Spliting datasets to match add_evidence requirement
        num_cols = alldata.shape[1]
        X = alldata[:, 0:num_cols - 1]
        Y = alldata[:, -1]
    return X, Y


def insample_RMSE_result(learner, train_x, train_y, verbose=False):
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    if verbose == True:
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
    return rmse


def outsample_RMSE_result(learner, test_x, test_y, verbose=False):
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    if verbose == True:
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
    return rmse

def insample_MAE_result(learner, train_x, train_y, verbose=False):
    pred_y = learner.query(train_x)  # get the predictions
    sum = abs(train_y - pred_y).sum()
    mae = sum / train_y.shape[0]
    if verbose == True:
        print()
        print("In sample results")
        print(f"MAE: {mae}")
    return mae

def outsample_MAE_result(learner, test_x, test_y, verbose=False):
    pred_y = learner.query(test_x)  # get the predictions
    sum = abs(test_y - pred_y).sum()
    mae = sum / test_y.shape[0]
    if verbose == True:
        print()
        print("Out of sample results")
        print(f"mae: {mae}")
    return mae

def calcute_RMSE(learners, num_leafs, train_x, train_y, test_x, test_y):
    in_sample_result = np.zeros((num_leafs, 1))
    out_sample_result = np.zeros((num_leafs, 1))
    for i in range(0, num_leafs):
        result = insample_RMSE_result(learners[i], train_x, train_y)
        in_sample_result[i] = result
        result = outsample_RMSE_result(learners[i], test_x, test_y)
        out_sample_result[i] = result
    return in_sample_result, out_sample_result

def calcute_MAE(learners, num_leafs, train_x, train_y, test_x, test_y):
    in_sample_result = np.zeros((num_leafs, 1))
    out_sample_result = np.zeros((num_leafs, 1))
    for i in range(0, num_leafs):
        result = insample_MAE_result(learners[i], train_x, train_y)
        in_sample_result[i] = result
        result = outsample_MAE_result(learners[i], test_x, test_y)
        out_sample_result[i] = result
    return in_sample_result, out_sample_result

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "czhang669"  # replace tb34 with your Georgia Tech username

if __name__ == "__main__":
    # if len(sys.argv) != 1:
    #   sys.exit(1)
    x, y = get_data(sys.argv[1])
    verbose = False
    np.random.seed(903614975)
    train_rows = int(0.6 * x.shape[0])
    test_rows = x.shape[0] - train_rows

    train_x = x[:train_rows, :]
    train_y = y[:train_rows]
    test_x = x[train_rows:, :]
    test_y = y[train_rows:]

    # experience 1
    num_leafs = 50
    learners = []
    # training
    for i in range(0, num_leafs):
        learners.append(dl.DTLearner(leaf_size=i + 1, verbose=verbose))
        learners[i].add_evidence(train_x, train_y)
    # testing
    in_sample_result, out_sample_result = calcute_RMSE(learners, num_leafs, train_x, train_y, test_x, test_y)

    fig, ax = plt.subplots()
    ax.set(title="Experiment 1: RMSE for In/Out-Sample Data of DTLearner under different leaf size"
           , xlabel='leaf size'
           , ylabel='RMSE')
    ax.plot(in_sample_result)
    ax.plot(out_sample_result)
    plt.legend(["in_sample_RMSE", "out_sample_RMSE"])
    plt.savefig("figure_1.png")

    # experience 2
    num_leafs = 50
    learners = []
    # training
    for i in range(0, num_leafs):
        learners.append(
            bl.BagLearner(learner=dl.DTLearner, kwargs={"leaf_size": i + 1, "verbose":False}, bags=20, boost=False, verbose=verbose))
        learners[i].add_evidence(train_x, train_y)
    # testing
    in_sample_result, out_sample_result = calcute_RMSE(learners, num_leafs, train_x, train_y, test_x, test_y)

    fig, ax = plt.subplots()
    ax.set(title="Experiment 2: RMSE for In/Out-Sample Data of BagLearner under different leaf size"
           , xlabel='leaf size'
           , ylabel='RMSE'
           , ylim=[0.0, 0.01])
    ax.plot(in_sample_result)
    ax.plot(out_sample_result)
    plt.legend(["in_sample_RMSE", "out_sample_RMSE"])
    plt.savefig("figure_2.png")

    # experience 3
    num_leafs = 50
    dt_learners = []
    rt_learners = []
    dt_train_time = []
    rt_train_time = []
    # training
    for i in range(0, num_leafs):
        dt_learners.append(dl.DTLearner(leaf_size=i + 1, verbose=verbose))
        start = time.time() * 1000
        dt_learners[i].add_evidence(train_x, train_y)
        end = time.time() * 1000
        dt_train_time.append(end - start)

    for i in range(0, num_leafs):
        rt_learners.append(rl.RTLearner(leaf_size=i + 1, verbose=verbose))
        start = time.time() * 1000
        rt_learners[i].add_evidence(train_x, train_y)
        end = time.time() * 1000
        rt_train_time.append(end - start)

    # testing
    dt_in_sample_result, dt_out_sample_result = calcute_MAE(dt_learners, num_leafs, train_x, train_y, test_x, test_y)
    rt_in_sample_result, rt_out_sample_result = calcute_MAE(rt_learners, num_leafs, train_x, train_y, test_x, test_y)

    fig, ax = plt.subplots()
    ax.set(title="Experiment 3: MAE for In/Out-Sample Data of DTLearner and LTLearner"
           , xlabel='leaf size'
           , ylabel='MAE'
           , ylim=[0.0, 0.01])
    ax.plot(dt_in_sample_result)
    ax.plot(dt_out_sample_result)
    ax.plot(rt_in_sample_result)
    ax.plot(rt_out_sample_result)
    plt.legend(["dt_in_sample_MAE", "dt_out_sample_MAE", "rt_in_sample_MAE", "rt_out_sample_MAE"])
    plt.savefig("figure_3.png")

    fig, ax = plt.subplots()
    ax.set(title="Experiment 3: trainingTime of DTLearner and LTLearner"
           , xlabel='leaf size'
           , ylabel='training time(ms)'
           )
    ax.plot(dt_train_time)
    ax.plot(rt_train_time)
    plt.legend(["dt_train_time", "rt_train_time"])
    plt.savefig("figure_4.png")
