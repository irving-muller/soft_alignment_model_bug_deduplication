"""
Implemented model proposed by Sun 2010
Sun, C., Lo, D., Wang, X., Jiang, J., & Khoo, S.-C. (2010).
A discriminative model approach for accurate duplicate bug report retrieval.
Proceedings of the 32nd ACM/IEEE International Conference on Software Engineering - ICSE ’10, 1, 45.
"""

class SunSVM(object):

    def __init__(self, bugDatabase):
        self.svm = None
        self.bugDatabase = bugDatabase

    def train(self, trainingPairs):








