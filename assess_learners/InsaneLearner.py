import BagLearner as bl
import LinRegLearner as lrl
import numpy as np
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.bagLearners = []
        for i in range(0, 20):
            learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose)
            self.bagLearners.append(learner)
    def add_evidence(self, x_train, y_train):
        for i in range(len(self.bagLearners)):
            self.bagLearners[i].add_evidence(x_train, y_train)
    def query(self, x_test):
        result = np.zeros((x_test.shape[0], len(self.bagLearners)))
        for i in range(len(self.bagLearners)):
            result[:, i] = self.bagLearners[i].query(x_test)
        return result.mean(axis=1)
    def author(self):
        return 'czhang669'
