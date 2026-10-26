import math
import numpy as np
from IPython.display import display

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from Visualization import state_analysis

class GroverSearch:
  def pqc(self, parameters, depth, n_qubits):
    expected_params = depth * n_qubits

    if len(parameters) != expected_params:
        raise ValueError(f"Expected {expected_params} parameters, but received {len(parameters)}. depth={depth}, n_qubits={n_qubits}")

    qc = QuantumCircuit(n_qubits)

    for d in range(depth):

        # Rotation layer
        for q in range(n_qubits):
            qc.ry(parameters[d * n_qubits + q], q)

        # Entangling layer only between rotation layers
        if d < depth - 1:

            # Even-indexed pairs
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)

            # Odd-indexed pairs
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)

    return qc

  def build_U(self):
    # U = (B† ⊗ I_I) A
    # Qubit order: |data>|index>
    # A encodes the database, B† acts on data qubits only

    value_qubits = QuantumRegister(self.n_value_q, name="value")
    data_qubits = QuantumRegister(self.n_feature_q, name="feature")
    index_qubits = QuantumRegister(self.n_index_q, name="index")

    qc = QuantumCircuit(value_qubits, data_qubits, index_qubits)

    A = self.pqc(self.A_parameters, self.depth_A, self.total_qubits)
    B = self.pqc(self.B_parameters, self.depth_B, (self.n_feature_q + self.n_value_q))

    # encode dataset
    qc.compose(A, qubits=list(range(self.total_qubits)), inplace=True)
    qc.barrier()

    # data elements which matches the query becomes all zero
    qc.compose(B.inverse(), qubits=(self.value_q + self.feature_q), inplace=True)
    qc.barrier()

    return qc

  def phase_flip_zero(self, qc, qubits):
    # Phase flip of |00...0> 
    # X gates convert |00...0> into |11...1>
    # H-MCX-H implements the multi-controlled Z operation to flip the phase of |11...1>

    if len(qubits) == 1:
        qc.x(qubits[0])
        qc.z(qubits[0])
        qc.x(qubits[0])

    else:
        qc.x(qubits)

        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

        qc.x(qubits)

    return qc

  def oracle(self): # inversion-test oracle Go
    # Go phase-flips the all-zero data state after B†
    # This makes the phase depend on the query-data overlap

    qc = QuantumCircuit(self.total_qubits)

    self.phase_flip_zero(qc, (self.value_q + self.feature_q))

    return qc

  def diffuser(self): # Gd = U† (phase flip of zero on all qubits) U
    # implements reflection about U|0>

    qc = QuantumCircuit(self.total_qubits)

    U = self.build_U()

    print("Applying inverse of U...")
    qc.compose(U.inverse(), qubits=self.total_q, inplace=True)
    qc.barrier()

    self.phase_flip_zero(qc, self.total_q)
    qc.barrier()

    qc.compose(U, qubits=self.total_q, inplace=True)

    return qc

  def run(self, dataset, query,
      A_parameters, B_parameters, depth_A=6, depth_B=3,
      num_of_iter=1, no_of_shots=2000, scheme="NEQR"
  ):
    # self.N_I, self.N_D = dataset.shape  # number of rows, number of columns/features
    self.N_I = len(dataset) # number of rows
    self.N_D = len(dataset[0])  # number of columns/features

    self.n_index_q = math.ceil(math.log2(self.N_I)) # represents row number or index of the data element
    self.n_feature_q = math.ceil(math.log2(self.N_D))  # represents feature number

    match scheme:
      case "NEQR":  # value represented in basis as |index>|feature>|value>
        max_value = int(np.max(dataset))
        self.n_value_q = max(1, math.ceil(math.log2(max_value + 1)))   # represents number of qubits required for the value of a feature of a data element
      case "FRQI":  # value represented in 1 qubit as intensity from |0> to |1> as angle theta
        self.n_value_q = 1
      case "AMPL":  # value represented as amplitude of |index>|feature>, no separate qubit required
        self.n_value_q = 0
      case _:
        raise ValueError("Use NEQR, FRQI, or AMPL.")

    self.total_qubits = self.n_index_q + self.n_feature_q + self.n_value_q 

    self.value_q = list(range(self.n_value_q))
    self.feature_q = list(range(self.n_value_q, (self.n_value_q + self.n_feature_q)))
    self.index_q = list(range((self.n_value_q + self.n_feature_q), self.total_qubits))
    self.total_q = list(range(self.total_qubits))

    # A and B parameters

    self.A_parameters = np.asarray(A_parameters)
    self.B_parameters = np.asarray(B_parameters)
    self.depth_A = depth_A
    self.depth_B = depth_B

    print("Number of data elements = ", self.N_I)
    print("Number of features = ", self.N_D)
    print("Index qubits = ", self.n_index_q)
    print("Data qubits = ", self.n_feature_q)
    print("Value qubits = ", self.n_value_q)
    print("Total no of qubits = ", self.total_qubits)
    print("Range of value qubits = ", self.value_q)
    print("Range of feature qubits = ", self.feature_q)
    print("Range of index qubits = ", self.index_q)

    # num_of_iter = max(1, round(((math.pi / 4) * math.sqrt(self.N_I)) - 0.5))
    # print("\nGrover iterations: ", num_of_iter)

    qc = self.build_U()
    state_analysis(Statevector.from_instruction(qc), "Encoded state U|Psi(0)>", self.n_value_q, self.n_feature_q, self.n_index_q)

    for iteration in range(num_of_iter):
        print(f"\n\n\nGROVER ITERATION {iteration + 1}")

        print("\n\n> Oracle: ")
        oracle_circuit = self.oracle()
        qc.compose(oracle_circuit, qubits=list(range(self.total_qubits)), inplace=True)
        state_analysis(Statevector.from_instruction(qc), "Oracle output", self.n_value_q, self.n_feature_q, self.n_index_q)

        print("\n\n> Diffuser: ")
        diffuser_circuit = self.diffuser()
        qc.compose(diffuser_circuit, qubits=list(range(self.total_qubits)), inplace=True)
        state_analysis(Statevector.from_instruction(qc), "Diffuser output", self.n_value_q, self.n_feature_q, self.n_index_q)

    print("\nFINAL MEASUREMENT")

    self.qc = qc.copy()

    classical_register = ClassicalRegister(self.n_index_q, name="index_result")
    self.qc.add_register(classical_register)

    self.qc.measure(self.index_q, classical_register)

    simulator = AerSimulator()
    compiled = transpile(self.qc, simulator)

    result = simulator.run(compiled, shots=no_of_shots).result()
    counts = result.get_counts()

    print(f"\nIndex measurement results for ({no_of_shots}) shots:")
    print(counts)

  def draw_circuit(self):
    display(self.qc.draw("mpl"))

  def analysis(self):
    print("REGISTER ANALYSIS")

    print("Dataset size (N):", self.N_I)
    print("Index qubits:", len(self.index_q))
    print("Data qubits:", len(self.feature_q))
    print("Total qubits:", self.qc.num_qubits)
    print("Classical bits:", self.qc.num_clbits)

    print("\nCIRCUIT ANALYSIS")

    print("Circuit width:", self.qc.width())
    print("Circuit size:", self.qc.size())
    print("Circuit depth/layers:", self.qc.depth())

    print("\nOPERATION COUNT")
    print(self.qc.count_ops())