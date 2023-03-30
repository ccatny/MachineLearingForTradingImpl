import math
import this

import numpy as np


class DTLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "czhang669"  # replace tb34 with your Georgia Tech username

    def str2float2int(self, num):
        return int(float(num))

    def select_factor(self, x_data, y_data):
        factor = 0
        coff = 0
        for i in range(x_data.shape[1]):
            if np.std(x_data[:, i]) == 0 or np.std(y_data) == 0:
                factor = 0
            else:
                current_coff = abs(np.corrcoef(x_data[:, i], y_data))
                # if np.std(x_data[:, i]) == 0, which means all the value are same, then we cannot use it since
                # the median and mean will be the same
                if current_coff[0, 1] >= coff and np.std(x_data[:, i]) != 0:
                    coff = current_coff[0, 1]
                    factor = i
        return factor

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size or np.std(data[:, -1]) == 0:
            sub_tree = np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])
            return sub_tree
        else:
            factor = self.select_factor(data[:, 0:-1], data[:, -1])
            split_val = np.median(data[:, factor])
            # Notice, if we have [1,2,2], the median will be 2 and if we use left <= split and right > split,
            # there will be infinite loop
            if split_val == np.max(data[:, factor]):
                split_val = np.mean(data[:, factor])
                # reference : https://blog.csdn.net/randompeople/article/details/104910146

            left_sub = self.build_tree(data[data[:, factor] <= split_val])
            right_sub = self.build_tree(data[data[:, factor] > split_val])

            #print(left_sub.shape)
            #print(right_sub.shape)
            root = np.array(([factor, split_val, 1, left_sub.shape[0] + 1],))
            return np.concatenate((root, left_sub, right_sub), axis=0)

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        data_y = np.resize(data_y, [data_x.shape[0], 1])
        data = np.concatenate((data_x, data_y), axis=1)
        self.tree = self.build_tree(data)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        y_test = np.zeros((points.shape[0],), dtype=float)
        for case in range(points.shape[0]):
            index = 0
            while index < self.tree.shape[0]:
                if self.tree[index][0] == 'leaf':
                    y_test[case] = float(self.tree[index][1])
                    break
                else:
                    factor = self.str2float2int(self.tree[index][0])
                    if points[case, factor] <= float(self.tree[index][1]):
                        index = index + self.str2float2int(self.tree[index][2])
                    else:
                        index = index + self.str2float2int(self.tree[index][3])
        return y_test


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
