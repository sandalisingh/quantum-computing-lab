import numpy as np
import random
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DiagonalGate

class GroverAdaptiveSearch:
    '''Use Grover search repeatedly to find the minima of a QUBO defined problem'''

    def __init__(self, Q, lambda_=1.2, constant=0):
        self.Q = np.asarray(Q, dtype=float)

        self.n = self.Q.shape[0]    # number of variables / qubits
        self.N = 2 ** self.n    # number of possible solutions
        self.no_of_iterations = int(math.ceil(math.log2(self.N)))  # maximum number of grover iterations

        self.lambda_ = lambda_
        self.constant = constant
        self.rng = random.Random(42) 

        # initialize random solution
        self.best_index = self.rng.randrange(self.N)
        self.best_solution = self.decode(self.best_index)   # selected path
        self.best_value = self.qubo(self.best_solution)     # cost of the path

        self.history = []

    def decode(self, index):
        return np.array([   # index to binary vector
            (index >> i) & 1
            for i in range(self.n)
        ])

    def qubo(self, x):  # calculate the cost of solution x
        return float(x @ self.Q @ x + self.constant)

    def oracle(self, threshold):
        phases = []

        for i in range(self.N):     # loop through all possible solutions - PROBLEM !!! - time complexity is O(2^n) - not scalable for large n
            x = self.decode(i)
            cost = self.qubo(x)

            if cost < threshold:    # change the phase of the solution if less than current candidate
                phases.append(-1)
            else:
                phases.append(1)

        return DiagonalGate(phases)

    def diffuser(self):
        # amplitude amplification
        qc = QuantumCircuit(self.n)

        qc.h(range(self.n)) # accumulates amplitudes in |00...0> state
        qc.x(range(self.n)) # |00...0> -> |11...1>

        # H-MCX-H gives MCZ - flips the sign of the |11...1> state
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)

        qc.x(range(self.n)) # |11...1> -> |00...0>
        qc.h(range(self.n))

        return qc

    def grover_circuit(self, threshold, iterations):
        qc = QuantumCircuit(self.n)

        qc.h(range(self.n)) # initialize superposition of all possible solutions

        oracle = self.oracle(threshold)
        diffuser = self.diffuser()

        for _ in range(iterations):
            qc.append(oracle, range(self.n))
            qc.compose(diffuser, inplace=True)

        return qc

    def measure(self, threshold, iterations):
        qc = self.grover_circuit(threshold, iterations)

        state = Statevector.from_instruction(qc)
        probabilities = state.probabilities()
        result = np.argmax(probabilities)  # chooses the solution with the highest probability

        return result

    def run(self, max_rounds=None):
        m = 1.0
        max_m = math.sqrt(self.N)   # max number of grover iterations

        print("Initial solution:", self.best_solution)
        print("Initial cost:", self.best_value)
        self.no_of_iterations = self.no_of_iterations if max_rounds is None else max_rounds # taking user input for max rounds if provided

        print("\n> Starting Grover Adaptive Search for", self.no_of_iterations, "rounds\n")

        for iteration in range(self.no_of_iterations):
            # randomly select grover iterations
            j = self.rng.randrange(max(1, math.ceil(m)))    # min 1 to avoid zero iterations

            candidate_index = self.measure(self.best_value, j)  # run grover search with current best value as threshold and j iterations

            candidate = self.decode(candidate_index)
            candidate_cost = self.qubo(candidate)

            improved = False

            if candidate_cost < self.best_value:
                self.best_index = candidate_index
                self.best_solution = candidate
                self.best_value = candidate_cost

                m = 1.0 
                improved = True

            else:
                m = min(self.lambda_ * m, max_m)

            current_info = {
                "Round": iteration + 1,
                "Grover_iterations": j,
                "Candidate": candidate.tolist(),
                "Candidate_cost": candidate_cost,
                "Best_solution": self.best_solution.tolist(),
                "Best_cost": self.best_value,
                "Improved": improved,
                "m": m
            }

            self.history.append(current_info)
            print(current_info)

        return self.best_solution, self.best_value