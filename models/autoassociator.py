import random

import numpy as np


class Autoassociator:
    def __init__(self, courses):
        n = courses.length()
        self.weights = np.zeros((n, n), dtype=np.int32)
        self.trainingCapacity = int(0.14 * (n - 1))
        self.refToCourses = courses
        self.offsetList = [29, 48, 90, 91, 92, 93, 94, 95, 96, 97, 144, 145, 147, 148, 161, 164, 171, 174, 176, 183, 184, 185, 187]
        self.random = random.Random()

    def getTrainingCapacity(self):
        return self.trainingCapacity

    def train(self, pattern):
        if len(pattern) == self.weights.shape[0] and self.trainingCapacity > 0:
            p = np.asarray(pattern, dtype=np.int32)
            outer = np.outer(p, p)
            np.fill_diagonal(outer, 0)
            self.weights += outer
            self.trainingCapacity -= 1

    def unitCourseSlotUpdate(self, slot):
        neurons = np.asarray(self.refToCourses.toPattern(slot), dtype=np.int32)
        W = self.weights
        for courseID in range(1, neurons.shape[0]):
            if neurons[courseID] == -1:
                total = int(W[courseID].dot(neurons))
                neurons[courseID] = 1 if total >= 0 else -1
            if neurons[courseID] == 1:
                self.refToCourses.setSlot(courseID, slot)

    def unitUpdate(self, neurons, index):
        arr = np.asarray(neurons, dtype=np.int32)
        total = int(self.weights[index].dot(arr))
        neurons[index] = 1 if total >= 0 else -1

    def unitUpdateRandom(self, neurons):
        index = 1 + self.random.randint(0, len(neurons) - 2)
        self.unitUpdate(neurons, index)
        return index

    def chainUpdate(self, neurons, steps):
        while steps > 0:
            self.unitUpdateRandom(neurons)
            steps -= 1
