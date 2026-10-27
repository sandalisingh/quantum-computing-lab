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
        bitstring = format(i, f"0{state.num_qubits}b")

        labels.append(bitstring)
        amplitudes.append(np.real(a))

    fig, ax = plt.subplots(figsize=(25, 6))
    ax.bar(labels, amplitudes)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set(
        title=title,
        xlabel="Basis State",
        ylabel="Signed Amplitude"
    )
    ax.tick_params(axis="x", labelrotation=90 if len(labels) > 10 else 0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

class oqmsa:
  def __init__(self):
    pass

  def state_preparation(self, dataset): # basis encoding
    N = len(dataset)
    max_element = max(dataset)
    self.n_data_q = math.log2(max_element)
    N_padded = 2**self.n_data_q
    
    qc = QuantumCircuit(self.n_data_q)
    amplitudes = np.zeros(2**self.n_data_q, dtype=complex)

    for data in dataset:
        amplitudes[data] = 1 / np.sqrt(N)

    state = Statevector(amplitudes)
    qc.initialize(state, range(self.n_data_q))
    qc.barrier()
    return qc

  def oracle(self, n_data_q, d, phi=math.pi):
    '''
    d : current minimum
    Oracle marks the phase of all the elements less than d
    '''
    # Create the oracle circuit
    oracle = QuantumCircuit(self.n_data_q, name="Oracle")

    # Find the bit patterns required to mark states from 0 to d
    binary_d = format(d, f"0{self.n_data_q}b")

    for k in range(self.n_data_q - 1, -1, -1): # examine each bit starting from msb towards lsb

        if d & (1 << k):  # if kth bit is 1


            # Create the prefix that identifies the states to mark
            prefix = binary_d[:self.n_data_q - k - 1] + "0"  # (a0 + a1 + ... a(k-1)) + "0"

            # Qubits corresponding to the prefix (MSB first)
            control_qubits = list(range(self.n_data_q - 1, self.n_data_q - len(prefix) - 1, -1))

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

  def diffuser(self, n_data_q, phi=math.pi):
    '''
    Diffuser amplifies the amplitude of all those elements marked by the oracle.
    '''
    diffuser = QuantumCircuit(n_data_q, name="Diffuser")

    # W† (inverse of state preparation)
    diffuser.h(range(n_data_q))

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
    diffuser.h(range(n_data_q))

    return diffuser

  def grover_long(self, no_of_iter, d):
    M = d
    phi = 2 * np.arcsin(np.sin(np.pi / np.sqrt(4 * no_of_iter + 2)) * np.sqrt(M / N))
    qc = QuantumCircuit(n_qubits, name="Grover-long")
    qc.compose(self.state_preparation(dataset), inplace=True)
    print("\n> Initial state before Grover-Long searching:")
    display(state.draw("latex"))
    plot_signed_amplitude(state, "Initial State")

    for i in range(no_of_iter):
      print("\n> Grover-Long iteration", i + 1)

      oracle = oracle(n_data_q, d, phi)
      qc.compile(oracle, inplace=True)
      print("\nState after applying the oracle:")
      state = Statevector.from_instruction(qc)
      display(state.draw("latex"))
      plot_signed_amplitude(state, f"After Oracle - Iteration {i + 1}")

      diffuser = diffuser(n_data_q, phi)
      qc.compile(diffuser, inplace=True)
      print("\nState after applying the diffuser:")
      state = Statevector.from_instruction(qc)
      display(state.draw("latex"))
      plot_signed_amplitude(state, f"After Diffuser - Iteration {i + 1}")

    print("\n> Grover-Long search completed with no of iterations =", no_of_iter)

    return qc

  def run(self, dataset):
    d = random.choice(dataset)  # current min chosen randomly
    n_data_q = (d + 1).bit_length()
    N = len(dataset)
    t_max = math.pi / 2 - np.arcsin(1/math.root(N)) / np.arcsin(1/math.root(N))
    t = 1 # initial t
    lambda_ = 6/5
    r = math.inf

    print("Initial minimum candidate =", d)
    print("Number of qubits =", n_qubits)
    print("Maximum Grover-Long iterations t_max =", t_max)

    # Create an equal superposition
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    state_before = Statevector.from_instruction(qc)
    state_before.draw("latex")

    for i in range(0, math.log2(N)):
      t_max = math.pi / 2 - np.arcsin(1/math.root(N)) / np.arcsin(1/math.root(N))
      M = d

      while t <= t_max and r > d:
        state_prep = state_preparation(dataset)
        qc.compile(state_prep, inplace=True)

        if M/N > 1/9:
          t_dash = random.randint(0, t)
          print("\n>> BHTT(GL) with no of iterations = ", t_dash)
          grover_long_iters = grover_long(t_dash)
          qc.compile(grover_long_iters, inplace=True)
          t = t * lambda_

        else:
          print("\n>> GL with no of iterations t_max = ", t_max)
          grover_long_iters = grover_long(t_max)
          qc.compile(grover_long_iters, inplace=True)

        final_state = Statevector.from_instruction(qc)
        print("Final state :")
        display(final_state.draw("latex"))
        plot_signed_amplitude(final_state, "Final State")

        probabilities = final_state.probabilities()

            measured_index = int(
                np.random.choice(
                    len(probabilities),
                    p=probabilities
                )
            )

            print("Measured basis-state index =", measured_index)

            # Convert the measured index to the corresponding dataset value
            r = dataset[measured_index]

            print("Measured dataset value r =", r)
            print("Current minimum candidate d' =", d_prime)

            # If the measured value is smaller, update the minimum
            if r < d_prime:

                print("\n[MINIMUM UPDATE]")
                print("New minimum found!")
                print("Previous minimum =", d_prime)
                print("New minimum =", r)

                d_prime = r

                # Reset the outer loop, as specified in Algorithm 1
                i = 0

                print("Reset outer iteration i = 0")

                break

            else:

                print("No smaller minimum found in this measurement.")

            # If using t_max, do not repeat the same search indefinitely
            if M / N <= 1 / 9:
                print("Completed the t_max search attempt.")
                break

        # Check whether a new minimum was found
        if r < d_prime:
            # Normally unreachable because d_prime was updated above;
            # retained for clarity when following Algorithm 1.
            i = 0
        else:
            i += 1

        print("\nEnd of outer iteration.")
        print("Current minimum =", d_prime)
        print("Next outer iteration i =", i)

        # Stop if no better candidate can be marked
        if sum(value < d_prime for value in dataset) == 0:
            print("\nNo smaller dataset items remain.")
            break

    print("OQMSA COMPLETED")

    return d_prime