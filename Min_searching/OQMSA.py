import math
import random
import numpy as np
from IPython.display import display

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, Diagonal
from qiskit_aer import AerSimulator

class OQMSA:
    """
    Optimized Quantum Minimum Searching Algorithm (OQMSA)
    N  = number of actual database items
    n  = number of qubits needed to encode the largest value
    Ne = 2^n = size of the quantum search space
    """

    def __init__(self, dataset):
        self.dataset = [int(x) for x in dataset]
        self.N = len(self.dataset)
        self.max_value = max(self.dataset)
        self.n = max(1, math.ceil(math.log2(self.max_value + 1)))
        self.Ne = 2 ** self.n
        
        self.rng = random.Random(42)
        self.backend = AerSimulator()
        self.history = []

        print("Dataset             : ", self.dataset)
        print("Number of items (N) : ", self.N)
        print("Data qubits (n)     : ", self.n)
        print("Quantum states (Ne) : ", self.Ne)

    def create_database_state(self):
        """
        Equal-amplitude quantum state:
        |psi> = 1/sqrt(Ne) sum_x |x>
        """

        amplitudes = np.zeros(self.Ne, dtype=complex)
        amplitudes[:] = 1 / np.sqrt(self.Ne)

        return amplitudes

    def create_W(self, state):
        """
        W |0...0> = |state>
        StatePreparation is used instead of initialize() because
        StatePreparation provides an invertible circuit.
        """

        W = QuantumCircuit(self.n, name="W")
        prep = StatePreparation(state)
        W.append(prep, range(self.n))

        return W

    def create_oracle(self, current_min):
        """
        Oracle for minimum search.

        Mark every basis state |x> satisfying :
            x <= current_min
        with a phase flip.
        """

        oracle = QuantumCircuit(self.n, name="Oracle")

        diagonal = np.ones(self.Ne, dtype=complex)

        for x in range(self.Ne):    # PROBLEM !!!
            if x <= current_min:
                diagonal[x] = -1

        oracle.append(Diagonal(diagonal), range(self.n))

        return oracle

    def phase_reversal(self, state, phi=np.pi):
        """
        W† I0 W
        """

        qc = QuantumCircuit(self.n, name="PhaseRev")
        prep = StatePreparation(state)  # statepreparation is invertible
        qc.append(prep.inverse(), range(self.n))  # W†

        # Apply phase to |0...0>.
        qc.x(range(self.n))
        if self.n == 1:
            qc.z(0)
        elif self.n == 2:
            qc.cz(0, 1)
        else:
            # H + MCX + H = MCZ
            qc.h(self.n - 1)
            qc.mcx(list(range(self.n - 1)), self.n - 1)
            qc.h(self.n - 1)
        qc.x(range(self.n))

        qc.append(prep, range(self.n))  # W

        return qc

    def grover_long_search(self, state, current_min, t_prime):
        """
        Iterate: W --> Oracle --> W† I0 W
        """

        qc = QuantumCircuit(self.n, name=f"GL_{t_prime}")
        prep = StatePreparation(state)

        # Initial W
        qc.append(prep, range(self.n))

        print("\n> Starting grover long iterations...")
        # Grover-Long iterations
        for iteration in range(t_prime):
            print(f"Iteration {iteration + 1}/{t_prime}")

            # Oracle
            print(">> Oracle...")
            oracle = self.create_oracle(current_min)
            qc.compose(oracle, range(self.n), inplace=True)

            # W† I0 W
            print(">> Diffuser...")
            phase_reversal = self.phase_reversal(state)
            qc.compose(phase_reversal, range(self.n), inplace=True)

        return qc

    def measure_circuit(self, circuit, shots=1):
        """
        Execute the circuit and return the measured integer.
        """

        measure_qc = circuit.copy()
        measure_qc.measure_all()
        compiled = transpile(measure_qc, self.backend)
        result = self.backend.run(compiled, shots=shots).result()
        counts = result.get_counts()
        print("Measurement counts: ", counts)

        measured_bitstring = max(counts, key=counts.get)
        measured_value = int(measured_bitstring, 2)
        print("Measured value: ", measured_value)
        return measured_value, counts

    def get_candidate_values(self, current_min):
        """
        Values from the actual database satisfying

                    d <= current_min

        This is useful for interpreting the quantum result.
        """

        return [
            d for d in self.dataset
            if d <= current_min
        ]


    def search(self, shots=1):
        """
        Execute the OQMSA search.
        Returns minimum
        """

        self.history = []
        current_min = self.rng.choice(self.dataset)

        print("\n> Starting OQMSA...")

        print("Initial candidate d' : ", current_min)

        state = self.create_database_state()
        lambda_ = 6 / 5

        max_outer_iterations = math.ceil(math.log2(self.N)) + 5
        print("Maximum outer iterations: ", max_outer_iterations)

        for outer_iteration in range(max_outer_iterations):
            M_hat = current_min + 1

            # t_max = floor(pi/(4*arcsin(sqrt(M/Ne))))
            ratio = min(max((M_hat / self.Ne), 1 / self.Ne), 1.0)
            theta = math.asin(math.sqrt(ratio))

            if theta == 0:
                t_max = 1
            else:
                t_max = max(1, math.floor(math.pi / (4 * theta)))

            t = 1
            improved = False

            print(f"\nOuter iteration {outer_iteration + 1}")
            print("Current d' : ",current_min)
            print("Estimated M : ", M_hat)
            print("t_max : ", t_max)

            while (t <= t_max and current_min > min(self.dataset)):
                # when estimated marked fraction is large,
                # choose a random t'.
                if ratio > 1 / 9:
                    upper = max(0, math.ceil(t))
                    t_prime = self.rng.randint(0, upper)
                else:
                    t_prime = t_max

                if t_prime == 0:    # do not use zero iterations
                    t_prime = 1

                print("  t =", t, " t' =", t_prime)
                
                search_circuit = (self.grover_long_search(state, current_min, t_prime))
                display(search_circuit.draw("mpl"))

                measured_value, counts = (self.measure_circuit(search_circuit, shots=shots))

                # The measured state is a candidate basis state.
                # Because the quantum search space contains
                # 0 ... Ne-1, a measured value may not correspond
                # to an actual database element.
                # Therefore we only accept it if it occurs in
                # the actual dataset.
                if measured_value in self.dataset:
                    candidate = measured_value
                else:
                    candidate = current_min

                print("  measured          : ", measured_value)
                print("  accepted candidate: ", candidate)

                if candidate < current_min:
                    old_min = current_min
                    current_min = candidate
                    improved = True
                    self.history.append(
                        {
                            "iteration": outer_iteration + 1,
                            "old_min": old_min,
                            "new_min": current_min,
                            "measured": measured_value,
                            "t_prime": t_prime
                        }
                    )
                    print("  d' changed: ", old_min, "->", current_min)

                    outer_iteration = 0
                    break

                t = t * lambda_

            if current_min == min(self.dataset):
                break

            if not improved:
                new_candidate = self.rng.choice(self.dataset)

                if new_candidate < current_min:
                    current_min = new_candidate
                    continue

                # If random candidate isn't smaller, use classical
                # candidate selection only to avoid an infinite loop.
                smaller_values = [
                    x for x in self.dataset
                    if x < current_min
                ]

                if smaller_values:
                    current_min = self.rng.choice(smaller_values)

                else:
                    break

        print("\n> Min searching finished.")

        print("\n>> Quantum minimum : ", current_min)
        print(">> Classical minimum: ", min(self.dataset))

        return current_min, self.history