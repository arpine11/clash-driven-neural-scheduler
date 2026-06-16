# Clash-Driven Neural Scheduler

A hybrid optimization framework that combines clash-driven propagation dynamics and Hopfield neural network repair for large-scale examination timetabling.

## Overview

Examination timetabling is a challenging NP-hard combinatorial optimization problem that requires assigning exams to timeslots while minimizing scheduling conflicts between students.

This project introduces a novel hybrid approach that combines:

* **Clash-Driven Propagation (CDP)** – a physics-inspired dynamic process that iteratively resolves conflicts through cyclic state transitions.
* **Hopfield Neural Network Repair** – an associative-memory mechanism that guides conflict-heavy schedules toward lower-conflict configurations.
* **Hybrid Scheduling Strategy** – alternating local dynamic optimization and global neural repair to escape local minima and improve convergence.

The framework was evaluated on all 12 Toronto Benchmark instances, one of the most widely used datasets for examination timetabling research.

---

## Key Features

* Hybrid optimization using dynamic propagation and neural repair
* Physics-inspired scheduling on a discrete toroidal state space
* Hopfield associative memory network with Hebbian learning
* Multiple exam ordering strategies
* Sensitivity and convergence analysis
* Reproducible experiments with fixed random seeds
* Evaluation on real-world benchmark datasets

---

## Research Contributions

### Clash-Driven Propagation

The scheduling process is modeled as a discrete dynamical system in which exams move across timeslots according to a conflict-driven shift operator.

The propagation parameter controls the exploration behavior of the search process, enabling:

* Quasi-periodic exploration
* Resonant dynamics
* Periodic-factor behavior
* Sensitivity analysis of scheduling trajectories

### Hopfield Network Repair

A Hopfield autoassociative neural network is trained on intermediate timetable states using Hebbian learning.

During optimization, the network performs global repair operations that:

* Move conflicting exams toward low-conflict patterns
* Reduce entrapment in local minima
* Improve solution quality on difficult benchmark instances
* Preserve improvements through rollback safeguards

---

## Experimental Results

The proposed approach was evaluated on the Toronto Benchmark Suite.

Key findings include:

* Dynamic-only optimization achieved zero clashes on 10 of 12 benchmark instances under standard timeslot settings.
* The hybrid approach outperformed the dynamic-only method on the most difficult minimum-timeslot scheduling scenarios.
* Sensitivity analysis revealed chaotic and stable scheduling regimes across benchmark instances.
* Rolling correlation analysis demonstrated that dynamic and hybrid approaches explore different regions of the search space.

---

## Project Structure

```text
data/               Toronto benchmark datasets
models/             Core data structures
scripts/            Experiment and evaluation scripts
figs/               Main visualizations
data_visuals/       Dataset analysis figures
sensitivity_figs/   Sensitivity analysis results
docs/               Documentation and reports
```

---

## Technologies

* Python 3.11
* NumPy
* Matplotlib
* Concurrent Processing
* Graph-Based Optimization
* Hopfield Neural Networks
* Scientific Computing

---

## Applications

While developed for examination timetabling, the framework can be extended to:

* University scheduling
* Workforce rostering
* Resource allocation
* Constraint satisfaction problems
* Large-scale combinatorial optimization

---

## Thesis

This repository accompanies the Bachelor of Science in Computer Science Capstone Thesis:

**"Clash-Driven Propagation with Hopfield Network Repair for Exam Timetabling Optimization"**

American University of Armenia, 2026

Author: **Arpine Tadevosyan**
