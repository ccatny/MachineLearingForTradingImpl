import numpy as np

def author():
    return 'czhang669'
class BagLearner(object):

    def __init__(self, learner=object, kwargs={}, bags=1, boost=False, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        self.boost = boost
        self.bags = bags
        #learner
        self.kwargs = kwargs
        self.bag_items = []
        for i in range(0, self.bags):
            self.bag_items.append(learner(**kwargs))

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "czhang669"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data_y = np.resize(data_y, [data_x.shape[0], ])
        # we cannot use concatenate here since data_y need to be reshaped to (n, 1) to use it,
        # however, this format cannot be used for LinRegLearner
        data = np.column_stack((data_y, data_y))
        size = data.shape[0]
        for i in range(self.bags):
            learner = self.bag_items[i]
            index = np.random.randint(low=0, high=size, size=size)
            select_x = data_x[index]
            select_y = data_y[index]
            learner.add_evidence(select_x, select_y)
        pass

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        result = np.empty((points.shape[0], self.bags))
        for i in range(self.bags):
            result[:, i] = self.bag_items[i].query(points)
        final = result.mean(axis=1)
        if self.verbose == True:
            print(final)
        return final

