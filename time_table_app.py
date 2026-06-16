import tkinter as tk
from tkinter import messagebox
from models.course_array import CourseArray
from models.autoassociator import Autoassociator


class TimeTableApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dynamic Time Table")
        self.geometry("600x800")

        self.canvas = tk.Canvas(self, width=400, height=700, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.controls = tk.Frame(self)
        self.controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.field_labels = ["Slots:", "Courses:", "Clash File:", "Iters:", "Shift:"]
        self.fields = []
        self.buttons = ["Load", "Start", "Cont", "StepAll", "Print", "Train", "Sens", "Exit"]

        self.courses = None
        self.hopfield = None

        self.iteration = 0
        self.max_iterations = 0
        self.min_clashes = float('inf')
        self.best_step = 0
        self.shift = 0
        self.running = False

        self.create_controls()
        self.initialize_defaults()

    def create_controls(self):
        for label in self.field_labels:
            tk.Label(self.controls, text=label).pack()
            entry = tk.Entry(self.controls)
            entry.pack()
            self.fields.append(entry)
        for text in self.buttons:
            btn = tk.Button(self.controls, text=text, command=lambda t=text: self.on_action(t))
            btn.pack(pady=2)

    def initialize_defaults(self):
        slots = 22
        default_values = [str(slots), "190", "./data/ear-f-83.stu", "1000", str(slots - 1)]
        for entry, value in zip(self.fields, default_values):
            entry.insert(0, value)

    def draw(self):
        slots = int(self.fields[0].get())
        width = slots * 10
        self.canvas.delete("all")
        create_line = self.canvas.create_line
        for course_index in range(1, self.courses.length()):
            clashes = self.courses.courseStatus(course_index)
            slot = self.courses.slot(course_index)
            color = "red" if clashes > 0 else "#90EE90"
            y = course_index
            x_slot = 10 * slot
            create_line(0, y, width, y, fill=color)
            create_line(x_slot, y, x_slot + 10, y, fill="black")

    def on_action(self, action):
        try:
            slots = int(self.fields[0].get())
            num_courses = int(self.fields[1].get())
            clash_file = self.fields[2].get()
            iters = int(self.fields[3].get())
            shift = int(self.fields[4].get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")
            return

        if action == "Load":
            self.courses = CourseArray(num_courses + 1, slots)
            self.courses.readClashes(clash_file)
            self.hopfield = Autoassociator(self.courses)
            self.draw()

        elif action == "Start":
            for i in range(1, self.courses.length()):
                self.courses.setSlot(i, 0)
            self.iteration = 1
            self.max_iterations = iters
            self.shift = shift
            self.min_clashes = float('inf')
            self.best_step = 0
            self.running = True
            self.schedule_iteration()

        elif action == "Cont":
            for iteration in range(1, iters + 1):
                self.courses.iterate(shift)
                self.draw()
                clashes = self.courses.clashesLeft()
                if clashes < self.min_clashes:
                    self.min_clashes = clashes
                    self.best_step = iteration
            print(f"Shift = {shift}, Min clashes = {self.min_clashes} at step {self.best_step}")
            self.courses.printSlotStatus()

        elif action == "StepAll":
            print("Offset\tClashes\tSlot0\tSlot(max)")
            max_courses = 0
            max_slot_id = 1
            for offset in range(self.courses.length() - 1):
                self.courses.iterateFrom(offset)
                self.draw()
                for i in range(1, slots):
                    stat = self.courses.slotStatus(i)
                    if stat[0] > max_courses:
                        max_courses = stat[0]
                        max_slot_id = i
                stat0 = self.courses.slotStatus(0)
                print(f"{offset}\t{stat0[1]}\t{stat0[0]}\t{max_courses}\t[{max_slot_id}]")

        elif action == "Print":
            print("Exam\tSlot\tClashes")
            for i in range(1, self.courses.length()):
                print(f"{i}\t{self.courses.slot(i)}\t{self.courses.courseStatus(i)}")

        elif action == "Train":
            for offset in self.hopfield.offsetList:
                self.courses.iterateFrom(offset)
                self.hopfield.train(self.courses.toPattern(1))
            print("Training complete.")

        elif action == "Sens":
            from engine import TimetableEngine
            from sensitivity import analyze, print_report
            engine = TimetableEngine(num_courses, slots, clash_file)
            trained = engine.train()
            print(f"Sensitivity probe (training patterns: {trained})")
            h_grid = [0, 2, 5, 10, 20]
            d_grid = sorted({max(1, iters // 4), max(1, iters // 2), iters})
            report = analyze(engine, h_grid, d_grid, shift)
            print_report(report)
            verdict = "SENSITIVE" if report['is_sensitive'] else "STABLE"
            messagebox.showinfo("Sensitivity", f"{verdict}\nnormalized = {report['normalized_sensitivity']:.3f}\nthreshold = {report['threshold']:.3f}\nSee console for full grid.")

        elif action == "Exit":
            self.destroy()

    def schedule_iteration(self):
        if not self.running:
            return
        if self.iteration > self.max_iterations:
            print(f"Minimum clashes {self.min_clashes} at step {self.best_step}")
            self.running = False
            return
        self.courses.iterate(self.shift)
        self.draw()
        clashes = self.courses.clashesLeft()
        print(f"Iteration {self.iteration}: Clashes = {clashes}")
        if clashes < self.min_clashes:
            self.min_clashes = clashes
            self.best_step = self.iteration
        if self.iteration % 50 == 0:
            for slot in range(self.courses.getPeriod() - 1, -1, -1):
                self.hopfield.unitCourseSlotUpdate(slot)
        self.iteration += 1
        self.after(1, self.schedule_iteration)


if __name__ == "__main__":
    app = TimeTableApp()
    app.mainloop()
