import math
import random
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator, Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, Diagonal
from qiskit_aer import AerSimulator


def plot_signed_amplitude(state, title):
    labels = []
    amplitudes = []

    for i, a in enumerate(state.data):
        if abs(a) > 1e-4:
            labels.append(i)
            amplitudes.append(np.real(a))

    fig, ax = plt.subplots(figsize=(25, 6))
    ax.bar(labels, amplitudes)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set(
        title=title,
        xlabel="Data (Decimal)",
        ylabel="Signed Amplitude"
    )
    ax.set_xticks(labels)
    ax.tick_params(axis="x", labelrotation=90 if len(labels) > 10 else 0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

class OQMSA:
    def __init__(self):
        pass

    def state_preparation(self, dataset): # basis encoding
        N = len(dataset)
        max_element = max(dataset)
        n_data_q = max(1, math.ceil(math.log2(max_element + 1)))
        N_padded = 2 ** n_data_q
        
        qc = QuantumCircuit(n_data_q)
        amplitudes = np.zeros(N_padded, dtype=complex)

        for data in dataset:
            amplitudes[data] = 1 
        amplitudes /= np.linalg.norm(amplitudes)

        state_prep = StatePreparation(amplitudes)
        qc.append(state_prep, range(n_data_q))
        qc.barrier()
        return qc

    def oracle(self, n_data_q, d, phi=np.pi):
        '''
        d : current minimum
        Oracle marks the phase of all the elements less than d
        '''
        # Create the oracle circuit
        oracle = QuantumCircuit(n_data_q, name="Oracle")

        # Find the bit patterns required to mark states from 0 to d
        binary_d = format(d, f"0{n_data_q}b")

        for k in range(n_data_q - 1, -1, -1): # examine each bit starting from msb towards lsb

            if d & (1 << k):  # if kth bit is 1


                # Create the prefix that identifies the states to mark
                prefix = binary_d[:n_data_q - k - 1] + "0"  # (a0 + a1 + ... a(k-1)) + "0"

                # Qubits corresponding to the prefix (MSB first)
                control_qubits = list(range(n_data_q - 1, n_data_q - len(prefix) - 1, -1))

                # Convert prefix zeros to ones
                for i, bit in zip(control_qubits, prefix):
                    if bit == "0":
                        oracle.x(i)

                # Apply phase e^(i*phi) to the matching states
                if len(control_qubits) == 1:
                    oracle.p(phi, control_qubits[0])
                else:
                    oracle.mcp(phi, control_qubits[:-1], control_qubits[-1])

                # Restore the original bit values
                for i, bit in zip(control_qubits, prefix):
                    if bit == "0":
                        oracle.x(i)

                oracle.barrier()
        return oracle

    def diffuser(self, dataset, n_data_q, phi=np.pi):
        '''
        Diffuser amplifies the amplitude of all those elements marked by the oracle.
        '''
        diffuser = QuantumCircuit(n_data_q, name="Diffuser")

        # W† (inverse of state preparation)
        prep = self.state_preparation(dataset)
        diffuser.compose(prep.inverse(), inplace=True)

        # I0: Apply phase e^(i*phi) to the |000...0> state

        # Convert |000...0> to |111...1>
        diffuser.x(range(n_data_q))

        # Apply phase e^(i*phi) when all qubits are 1
        if n_data_q == 1:
            diffuser.p(phi, 0)
        else:
            diffuser.mcp(phi, list(range(n_data_q - 1)), n_data_q - 1)

        # Restore the original bit values
        diffuser.x(range(n_data_q))

        # W (state preparation)
        diffuser.compose(prep, inplace=True)

        return diffuser

    def grover_long(self, dataset, n_data_q, no_of_iter, d):
        qc = QuantumCircuit(n_data_q, name="Grover-long")
        qc.compose(self.state_preparation(dataset), inplace=True)
        print("\n> Initial state before Grover-Long searching:")
        state = Statevector.from_instruction(qc)
        display(state.draw("latex"))
        plot_signed_amplitude(state, "Initial State")

        oracle = self.oracle(n_data_q, d)
        diffuser = self.diffuser(dataset, n_data_q)

        for i in range(no_of_iter):
            print("\n> Grover-Long iteration", i + 1)

            qc.compose(oracle, inplace=True)
            state = Statevector.from_instruction(qc)
            display(state.draw("latex"))
            plot_signed_amplitude(state, f"After Oracle - Iteration {i + 1}")

            qc.compose(diffuser, inplace=True)
            state = Statevector.from_instruction(qc)
            display(state.draw("latex"))
            plot_signed_amplitude(state, f"After Diffuser - Iteration {i + 1}")

        return qc

    def run(self, dataset):
        """
        measured value replaces the current candidate when
        the measured value is smaller
        """

        d = random.choice(dataset)  # current min chosen randomly

        state_prep = self.state_preparation(dataset)

        max_element = max(dataset)
        n_data_q = max(1, math.ceil(math.log2(max_element + 1)))
        N = len(dataset)
        backend = AerSimulator()

        print("Dataset = ", dataset)
        print("Number of data qubits = ", n_data_q)
        print("\n>> Initial minimum candidate = ", d)

        i = 0
        while i < (math.ceil(math.log2(N)) + 1):
            theta = np.arcsin(1 / np.sqrt(N))
            t_max = int((math.pi / 2 - theta) / theta)
            # M = d
            M = sum(1 for value in dataset if value < d)
            t = 1 # initial t
            lambda_ = 6/5
            r = math.inf

            while t <= t_max and r > d:
                qc = QuantumCircuit(n_data_q)

                if M/N > 1/9:
                    t_dash = int(random.randint(0, t))
                    print("\n>> BHTT(GL) with no of iterations = ", t_dash)
                    grover_long_iters = self.grover_long(dataset, n_data_q, t_dash, d)
                    qc.compose(grover_long_iters, inplace=True)
                    t = t * lambda_
                else:
                    print("\n>> GL with no of iterations t_max = ", t_max)
                    grover_long_iters = self.grover_long(dataset, n_data_q, t_max, d)
                    qc.compose(grover_long_iters, inplace=True)

                print("\n>> Measuring...")
                meas_qc = qc.copy()
                meas_qc.measure_all()

                meas_qc = transpile(meas_qc, backend)
                job = backend.run(meas_qc, shots=1, memory=True)
                result = job.result()
                r = int(result.get_memory()[0], 2)

                print(f"Measured value: {r}")

            if r not in dataset:
                print("Measured value not in dataset")
            elif r < d:
                print("\n> MINIMUM UPDATE")
                print("New candidate = ", r)
                d = r
                i = 0
            else:
                print("No smaller candidate found in this round.")
                i += 1

        print("\n> OQMSA completed.")
        print(">> Minimum candidate found =", d)

        return d
