from models.course_array import CourseArray
from models.autoassociator import Autoassociator


class TimetableEngine:
    def __init__(self, num_courses, slots, clash_file):
        self.num_courses = num_courses
        self.slots = slots
        self.clash_file = clash_file
        self.courses = CourseArray(num_courses + 1, slots)
        self.courses.readClashes(clash_file)
        self.hopfield = Autoassociator(self.courses)
        self._trained = False

    def train(self, count=None):
        offsets = self.hopfield.offsetList
        if count is not None:
            offsets = offsets[:count]
        for offset in offsets:
            self.courses.iterateFrom(offset)
            self.hopfield.train(self.courses.toPattern(1))
        self._trained = True
        return len(offsets)

    def reset_slots(self):
        for i in range(1, self.courses.length()):
            self.courses.setSlot(i, 0)

    def slot_assignment(self):
        return tuple(self.courses.slot(i) for i in range(1, self.courses.length()))

    def run(self, dynamic_steps, hopfield_runs, shift, *, reset=True, record_history=False):
        if reset:
            self.reset_slots()
        if hopfield_runs > 0 and dynamic_steps > 0:
            period = max(1, dynamic_steps // hopfield_runs)
        else:
            period = dynamic_steps + 1
        history = [] if record_history else None
        min_clashes = float('inf')
        best_step = 0
        applied = 0
        final_clashes = self.courses.clashesLeft()
        for step in range(1, dynamic_steps + 1):
            self.courses.iterate(shift)
            clashes = self.courses.clashesLeft()
            final_clashes = clashes
            if history is not None:
                history.append(clashes)
            if clashes < min_clashes:
                min_clashes = clashes
                best_step = step
            if applied < hopfield_runs and step % period == 0:
                for slot in range(self.courses.getPeriod() - 1, -1, -1):
                    self.hopfield.unitCourseSlotUpdate(slot)
                applied += 1
        return {
            'dynamic_steps': dynamic_steps,
            'hopfield_runs': hopfield_runs,
            'hopfield_applied': applied,
            'shift': shift,
            'min_clashes': min_clashes if min_clashes != float('inf') else final_clashes,
            'best_step': best_step,
            'final_clashes': final_clashes,
            'slot_assignment': self.slot_assignment(),
            'history': history,
        }
