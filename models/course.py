class Course:
    def __init__(self, newSlot=0):
        self.clashesWith = []
        self.mySlot = newSlot
        self.force = 0

    def addClash(self, thatClash):
        self.clashesWith.append(thatClash)

    def clashSize(self):
        result = 0
        for i in range(len(self.clashesWith)):
            if self.mySlot == self.clashesWith[i].mySlot:
                result += 1
        return result

    def unitClashForce(self):
        for i in range(len(self.clashesWith)):
            if self.mySlot == self.clashesWith[i].mySlot:
                return 1
        return 0

    def setForce(self):
        self.force = self.unitClashForce()

    def shift(self, limit):
        self.mySlot += self.force
        if self.mySlot < 0:
            self.mySlot = limit - 1
        elif self.mySlot >= limit:
            self.mySlot = 0
