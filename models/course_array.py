from models.course import Course

class CourseArray:
    def __init__(self, numOfCourses, numOfSlots):
        self.setPeriod(numOfSlots)
        self.elements = [None] * numOfCourses
        for i in range(1, len(self.elements)):
            self.elements[i] = Course()

    def readClashes(self, filename):
        try:
            with open(filename, 'r') as file:
                line = file.readline()
                while line:
                    tokens = line.strip().split()
                    count = len(tokens)
                    if count > 1:
                        index = [int(token) for token in tokens]
                        for i in range(len(index)):
                            for j in range(len(index)):
                                if j != i:
                                    k = 0
                                    while k < len(self.elements[index[i]].clashesWith) and self.elements[index[i]].clashesWith[k] != self.elements[index[j]]:
                                        k += 1
                                    if k == len(self.elements[index[i]].clashesWith):
                                        self.elements[index[i]].addClash(self.elements[index[j]])
                    line = file.readline()
        except Exception:
            pass

    def length(self):
        return len(self.elements)

    def getPeriod(self):
        return self.period

    def setPeriod(self, numOfSlots):
        self.period = numOfSlots

    def courseStatus(self, index):
        return self.elements[index].clashSize()

    def slot(self, index):
        return self.elements[index].mySlot

    def setSlot(self, index, newSlot):
        self.elements[index].mySlot = newSlot

    def maxClashSize(self, index):
        return 0 if self.elements[index] is None or not self.elements[index].clashesWith else len(self.elements[index].clashesWith)

    def clashesLeft(self):
        result = 0
        for i in range(1, len(self.elements)):
            result += self.elements[i].clashSize()
        return result

    def iterate(self, shifts):
        for index in range(1, len(self.elements)):
            self.elements[index].setForce()
            for move in range(1, shifts + 1):
                if self.elements[index].force != 0:
                    self.elements[index].setForce()
                    self.elements[index].shift(self.period)

    def iterateFrom(self, offset):
        for index in range(1, len(self.elements)):
            self.setSlot(index, 0)

        length = len(self.elements) - 1
        for index in range(length):
            id = (index + offset) % length + 1
            self.elements[id].setForce()
            for move in range(1, self.period + 1):
                if self.elements[id].force != 0:
                    self.elements[id].setForce()
                    self.elements[id].shift(self.period)

    def slotStatus(self, slot):
        result = [0, 0]
        for i in range(1, len(self.elements)):
            if self.elements[i].mySlot == slot:
                result[0] += 1
                result[1] += self.elements[i].clashSize()
        return result

    def toPattern(self, slot):
        result = [0] * len(self.elements)
        for i in range(1, len(result)):
            result[i] = 1 if self.elements[i].mySlot == slot else -1
        return result

    def printSlotStatus(self):
        for slot in range(self.period):
            status = self.slotStatus(slot)
            print(f"{slot}\t{status[0]}\t{status[1]}")

    def printResult(self):
        for i in range(1, len(self.elements)):
            print(f"{i}\t{self.elements[i].mySlot}")
