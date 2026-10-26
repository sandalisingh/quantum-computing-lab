import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from Visualization import compare_similarlity_bw_two_states, state_analysis

class ApproximateEncoder:
    # used to generate operation A and B
    # A is the unitary operation used to approximately encode the dataset
    # B is the unitary operation used to approximately encode the query

    def __init__(self):
        self.rng = np.random.default_rng(42)

    # ----------- ENCODING SCHEME -----------

    def prepare_neqr_dataset_target(self, dataset):
        dataset = np.array(dataset, dtype=float, copy=True)
        self.N_I, self.N_D = dataset.shape
        self.n_index_q = math.ceil(math.log2(self.N_I))
        self.n_feature_q = math.ceil(math.log2(self.N_D))

        self.min_value = 0
        self.max_value = int(np.max(dataset))
        self.n_value_q = max(1, math.ceil(math.log2(self.max_value + 1)))

        total_qubits = self.n_index_q + self.n_feature_q + self.n_value_q
        total_num_states = 2**total_qubits

        print(f"\nNum of data elements (N_I): {self.N_I}")
        print(f"Num of features (N_D): {self.N_D}")
        print(f"Number of index qubits (n_index_q): {self.n_index_q}")
        print(f"Number of data qubits (n_data_q): {self.n_feature_q}")
        print(f"Number of value qubits (n_value_q): {self.n_value_q}")
        print(f"Total number of qubits (total_qubits): {total_qubits}")
        print("Total number of states: ", total_num_states)

        # Statevector dimension
        target = np.zeros(total_num_states, dtype=complex)  

        # Equal amplitude for each padded index pair
        amplitude = 1 / np.sqrt(self.N_I * self.N_D)
        print("\nAmplitude = ", amplitude)

        for i in range(self.N_I): # O(N_I x N_D)
            for j in range(self.N_D):
                value = int(dataset[i, j])

                # Quantum state = |i > |j> |value>
                index = (
                    value       # value of a feature at a given row
                    + (j << self.n_value_q)  # feature
                    + (i << (self.n_value_q + self.n_feature_q)) # index
                )

                target[index] = amplitude
                # print(f"Value: {value}/{value:b}; i={i} j={j}; Index: {index}/{index:b}; Target amplitude: {target[index]}")

        return Statevector(target)

    def prepare_neqr_query_target(self, query):
        query = np.array(query, dtype=int, copy=True)

        total_qubits = self.n_value_q + self.n_feature_q
        total_num_states = 2**total_qubits

        print(f"\nNumber of feature qubits (n_data_q): {self.n_feature_q}")
        print(f"Number of value qubits (n_value_q): {self.n_value_q}")
        print(f"Total number of qubits (total_qubits): {total_qubits}")
        print(f"Total number of states (total_num_states): {total_num_states}")

        target = np.zeros(total_num_states, dtype=complex)

        # Uniform amplitude over features
        amplitude = 1 / np.sqrt(self.N_D)
        print("\nAmplitude = ", amplitude)

        for j, value in enumerate(query):
            # Quantum state = |j> |value>
            index = value + (int(j) << self.n_value_q)
            target[index] = amplitude

            # print(f"Value: {value}/{value:b}; j={j}; Index: {index}/{index:b}; Target amplitude: {target[index]}")

        return Statevector(target)

    def prepare_frqi_dataset_target(self, dataset):
        """
        State = 1/sqrt(N_I*N_D) * sum_i sum_j [ |i> |j> [cos(theta_ij)|0> + sin(theta_ij)|1>]]
        = |record index>|feature index>|intensity>
        """

        dataset = np.asarray(dataset, dtype=float)

        self.N_I, self.N_D = dataset.shape
        self.min_value = 0.0
        self.max_value = float(np.max(dataset))

        # Qubits for record and feature indices
        self.n_index_q = math.ceil(math.log2(self.N_I))
        self.n_feature_q = math.ceil(math.log2(self.N_D))

        # FRQI uses one intensity qubit
        self.n_value_q = 1

        total_qubits = self.n_index_q + self.n_feature_q + self.n_value_q

        print(f"\nNumber of record-index qubits: {self.n_index_q}")
        print(f"Number of feature-index qubits: {self.n_feature_q}")
        print(f"Number of intensity qubits: {self.n_value_q}")
        print(f"Total number of qubits: {total_qubits}")
        print("Total number of states:", 2**total_qubits)

        # Map values to [0, 1], then to angles [0, pi/2]
        normalized = (dataset - self.min_value) / (self.max_value - self.min_value)
        theta = (np.pi / 2) * normalized

        target = np.zeros(2**total_qubits, dtype=complex)
        amplitude = 1 / np.sqrt(self.N_I * self.N_D)

        for i in range(self.N_I):
            for j in range(self.N_D):
                angle = theta[i, j]

                # Index of |index i> |feature j> |intensity=0>
                index_0 = (i << (self.n_feature_q + self.n_value_q)) + (j << self.n_value_q)

                # Index of |index i> |feature j> |intensity=1>
                index_1 = index_0 + 1

                target[index_0] = amplitude * np.cos(angle) # cosθ |0>
                target[index_1] = amplitude * np.sin(angle) # sinθ |1>

                print(f"dataset[{i},{j}] = {dataset[i,j]}")
                print(f"index_0 = {index_0}; amplitude (cos) = {target[index_0]}")
                print(f"index_1 = {index_1}; amplitude (sin) = {target[index_1]}")

        return Statevector(target)

    def prepare_frqi_query_target(self, query):
        """
        State = 1/sqrt(N_D) * sum_j [ |j> [cos(theta_j)|0> + sin(theta_j)]|1>]
        = |intensity>|feature index>
        """

        query = np.asarray(query, dtype=float).flatten()
        total_qubits = self.n_feature_q + self.n_value_q

        print(f"\nNumber of feature-index qubits: {self.n_feature_q}")
        print(f"Number of intensity qubits: {self.n_value_q}")
        print(f"Total number of qubits: {total_qubits}")
        print("Total number of states:", 2**total_qubits)

        normalized = (query - self.min_value) / (self.max_value - self.min_value)
        theta = (np.pi / 2) * normalized

        target = np.zeros(2**total_qubits, dtype=complex)
        amplitude = 1 / np.sqrt(self.N_D)

        for j in range(self.N_D):
            angle = theta[j]

            # |feature j> |intensity=0>
            index_0 = j << self.n_value_q

            # |feature j> |intensity=1>
            index_1 = index_0 + 1

            target[index_0] = amplitude * np.cos(angle) # cosθ |0>
            target[index_1] = amplitude * np.sin(angle) # sinθ |1>

        return Statevector(target)

    def prepare_amplitude_dataset_target(self, dataset):
        dataset = np.array(dataset, dtype=float, copy=True)
        self.N_I, self.N_D = dataset.shape

        self.n_index_q = math.ceil(math.log2(self.N_I))
        self.n_feature_q = math.ceil(math.log2(self.N_D))
        self.n_value_q = 0

        padded_N_I = 2 ** self.n_index_q
        padded_N_D = 2 ** self.n_feature_q

        total_qubits = self.n_index_q + self.n_feature_q

        print(f"\nNumber of index qubits: {self.n_index_q}")
        print(f"Number of data qubits: {self.n_feature_q}")
        print(f"Total number of qubits: {total_qubits}")
        print(f"Padded number of index states: {padded_N_I}")
        print(f"Padded number of data states: {padded_N_D}")
        print("Total number of states:", padded_N_I * padded_N_D)

        print(f"\nOriginal dataset: {dataset}")

        # Normalize each nonzero row
        norms = np.linalg.norm(dataset, axis=1, keepdims=True)  # axis=1 works on each row, computes L2 euclidean norm = summation x^2; returns (N,1) matrix
        # normalized_dataset = np.zeros_like(dataset, dtype=float)    # create a zero matrix of same shape as dataset

        # nonzero_rows = norms[:, 0] > 0
        # normalized_dataset[nonzero_rows] = (
        #     dataset[nonzero_rows] / norms[nonzero_rows]
        # )

        # # Assign equal weight to every row, including zero rows.
        # # A zero row is represented by a uniform feature state.
        # normalized_dataset[~nonzero_rows] = 1 / np.sqrt(self.N_D)

        divisor = np.sqrt(np.sum(dataset ** 2))
        if divisor == 0:
            raise ValueError("Cannot normalize a dataset containing only zeros.")
        normalized_dataset = dataset / divisor

        print("Normalized dataset = ")
        print(normalized_dataset)

        target = np.zeros(padded_N_I * padded_N_D, dtype=complex)

        for i in range(self.N_I):
            for j in range(self.N_D):
                target[i * padded_N_D + j] = (
                    normalized_dataset[i, j]
                )

        target_norm = np.linalg.norm(target)
        assert np.isclose(target_norm, 1.0), f"Target is not normalized: {target_norm}"

        print(f"\nTarget: {target}")
        print(f"Target norm: {target_norm}")

        return Statevector(target)

    def prepare_amplitude_query_target(self, query):
        query = np.array(query, dtype=float, copy=True)
        
        N_Q = len(query)
        n_query_q = math.ceil(math.log2(N_Q))
        total_qubits = n_query_q
        padded_N_Q = 2 ** n_query_q
        print(f"\nNumber of query qubits: {n_query_q}")
        print(f"Total number of qubits: {total_qubits}")
        print(f"Padded number of query states: {padded_N_Q}")

        # create target

        print(f"\nOriginal query: {query}")

        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("Query has zero norm; it cannot be amplitude encoded.")

        query = query / norm

        target = np.zeros(padded_N_Q, dtype=float)
        target[:N_Q] = query

        if not np.isclose(np.linalg.norm(target), 1.0):
            raise ValueError("Query target is not normalized.")

        print(f"\nTarget: {target}")

        return Statevector(target)

    # ----------- GRADIENT, LOSS AND OPTIMIZATION -----------

    def walsh_hadamard_transform(self, state):
        state = np.asarray(state, dtype=complex).copy()
        N_D = len(state)

        if N_D == 0 or (N_D & (N_D - 1)) != 0:
            raise ValueError("State-vector length must be a power of 2.")

        step = 1

        while step < N_D:
            for start in range(0, N_D, 2 * step):
                a = state[start:start + step].copy()
                b = state[start + step:start + 2 * step].copy()

                state[start:start + step] = a + b
                state[start + step:start + 2 * step] = a - b

            step *= 2

        return state / np.sqrt(N_D)

    def gaussian_kernel(self, N, gamma=64):
        """
        k(i,j) = exp(-64 * (i-j)^2)

        The kernel is defined over outcome indices.
        Returns an N x N kernel matrix.
        """

        indices = np.arange(N, dtype=float)

        differences = indices[:, None] - indices[None, :]

        return np.exp(-gamma * differences**2)

    def mmd_loss(self, P, Q, kernel):
        """
        Squared MMD between two discrete probability distributions:
        MMD^2(P,Q) = P.T @ K @ P + Q.T @ K @ Q - 2 * P.T @ K @ Q
        """

        P = np.asarray(P, dtype=float).ravel()  # converts to 1D array of floats
        Q = np.asarray(Q, dtype=float).ravel()

        if P.shape != Q.shape:
            raise ValueError("P and Q must have the same length.")

        if kernel.shape != (len(P), len(P)):
            raise ValueError("Kernel shape must match distribution length.")

        # numpy automatically handles the transpose
        # when provided with a 1d vector
        # numpy automatically treats as a row vector when its on the left 
        # and a column vector when its on the right
        loss = ((P @ kernel @ P) + (Q @ kernel @ Q) - (2 * P @ kernel @ Q))

        # Avoid tiny negative values caused by floating-point rounding.
        return max(float(loss), 0.0)

    def mmd_cost_function(self, parameters, target_amplitudes, kernel, depth, n_qubits):
        """
        Calculate L = (L1 + L2) / 2.

        L1 compares computational-basis probabilities.
        L2 compares probabilities after Walsh-Hadamard transform.
        """
        qc = self.pqc(parameters, depth, n_qubits)
        state = Statevector.from_instruction(qc)
        amplitudes = np.real_if_close(state.data).real

        # L1: compare computational-basis probabilities.
        probability_1 = np.abs(amplitudes) ** 2
        target_probability_1 = np.abs(target_amplitudes) ** 2

        L1 = self.mmd_loss(probability_1, target_probability_1, kernel)

        # L2: compare probabilities after applying H to every qubit.
        transformed_amplitudes = self.walsh_hadamard_transform(amplitudes)
        target_transformed_amplitudes = self.walsh_hadamard_transform(target_amplitudes)

        probability_2 = np.abs(transformed_amplitudes) ** 2
        target_probability_2 = np.abs(target_transformed_amplitudes) ** 2

        L2 = self.mmd_loss(probability_2, target_probability_2, kernel)

        L = (L1 + L2) / 2

        target_state = Statevector(target_amplitudes)
        fidelity = state_fidelity(state, target_state)

        return L, fidelity

    def fidelity_cost_function(self, parameters, target_amplitudes, depth, n_qubits):
        qc = self.pqc(parameters, depth, n_qubits)
        state = Statevector.from_instruction(qc)
        target_state = Statevector(target_amplitudes)
        fidelity = state_fidelity(state, target_state)
        
        return (1-fidelity), fidelity
 
    def parameter_shift_gradient(self, parameters, objective):
        """
        Calculate the MMD loss gradient using the parameter-shift rule:
        grad = (L(theta + pi/2) - L(theta - pi/2)) / 2
        """
        grad = np.zeros_like(parameters)
        shift = np.pi / 2

        for i in range(len(parameters)):    # O(depth * n_qubits)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[i] += shift
            minus[i] -= shift

            # Evaluate the full cost function L at the shifted parameters
            # (Note: cost_function returns tuple: (L, fidelity))
            L_plus, _ = objective(plus)
            L_minus, _ = objective(minus)

            # Apply the parameter-shift rule formula
            grad[i] = (L_plus - L_minus) / 2

        return grad

    def adam_optimizer(self, objective, initial_parameters, epochs, learning_rate=0.005):
        print("\n> Using adam optimization....")

        parameters = initial_parameters.copy()
        m = np.zeros_like(parameters) # creates an array of zeros with the same shape as parameters
        v = np.zeros_like(parameters)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        loss_history = []
        fidelity_history = []
        best_parameters = initial_parameters.copy()

        best_fidelity = -float("inf")
        patience = 10 # number of epochs to wait for improvement before stopping
        min_delta = 1e-6 # minimum change in the loss to qualify as an improvement
        wait = 0

        for epoch in range(1, epochs + 1):
            gradient = self.parameter_shift_gradient(parameters, objective)

            m = (beta1 * m) + ((1 - beta1) * gradient)
            v = (beta2 * v) + ((1 - beta2) * (gradient**2))

            m_hat = m / (1 - beta1**epoch)
            v_hat = v / (1 - beta2**epoch)

            # lr = learning_rate if epoch <= 100 else learning_rate/10
            parameters -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            loss, fidelity = objective(parameters)
            loss_history.append(loss)
            fidelity_history.append(fidelity)

            # print(f"Gradient ({gradient.shape}) = ", gradient)
            print(f"Epoch {epoch} - Loss {loss} - Fidelity {fidelity}")

            if fidelity > best_fidelity + min_delta:
                best_fidelity = fidelity
                best_parameters = parameters.copy()
                wait = 0
            else:
                wait += 1

            # early stopping if the loss does not improve for a certain number of epochs

            # if wait >= patience:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

        epochs = range(1, len(loss_history) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: MMD Loss
        ax1.plot(epochs, loss_history, color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MMD Loss')
        ax1.set_title('MMD Loss')
        ax1.grid(True)

        # Subplot 2: Fidelity Loss
        ax2.plot(epochs, fidelity_history, color='tab:orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Fidelity Loss')
        ax2.set_title('Fidelity')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        return best_parameters, best_fidelity

    # ----------- CIRCUIT GENERATION AND TRAINING -----------
    
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

    def train_circuit(self, target_amplitudes, depth, epochs, learning_rate=0.1, loss_function_type="FIDELITY"):
        """Train a parameterized circuit to approximate a target state."""
        print("\n> Starting training....")

        if isinstance(target_amplitudes, Statevector):
            target_amplitudes = target_amplitudes.data
        target_amplitudes = np.asarray(target_amplitudes, dtype=complex).real

        n_qubits = int(round(math.log2(len(target_amplitudes))))

        num_parameters = depth * n_qubits
        initial_parameters = self.rng.uniform(-np.pi, np.pi, num_parameters)
        print(f"\nNumber of parameters = {num_parameters}")
        # print(f"Initial parameters = {initial_parameters}")

        kernel = None

        match loss_function_type:
            case "MMD":
                kernel = self.gaussian_kernel(len(target_amplitudes))
                plt.imshow(kernel, cmap='viridis')
                plt.colorbar()
                plt.title(f'Kernel {kernel.shape}')
                plt.show()
                def objective(parameters):
                    loss, fidelity = self.mmd_cost_function(
                        parameters, target_amplitudes, kernel,
                        depth, n_qubits
                    )
                    return loss, fidelity
            case _:
                def objective(parameters):
                    loss, fidelity = self.fidelity_cost_function(
                        parameters, target_amplitudes, depth, n_qubits
                    )
                    return loss, fidelity
                
        
        trained_parameters, best_fidelity = self.adam_optimizer(
            objective, initial_parameters, epochs, learning_rate
        )

        trained_circuit = self.pqc(trained_parameters, depth, n_qubits)
        
        print(f"\n> Training finished.")
        print(f"Number of qubits: {n_qubits}")
        print(f"Circuit depth (layers): {depth}")
        print(f"Number of parameters: {num_parameters}")
        print(f"Best fidelity achieved: {best_fidelity:.8f}")

        return trained_parameters, trained_circuit, best_fidelity

    def generate_encoding(self, data, scheme="NEQR", depth=3, epochs=300, learning_rate=0.1, loss_function_type="FIDELITY"):
        data = np.asarray(data)

        if data.ndim == 2:
            print("\n> Preparing dataset target state....")
            suffix = "dataset_target"
        elif data.ndim == 1:
            print("\n> Preparing query target state....")
            suffix = "query_target"
        else:
            raise ValueError("Data must be a 1D query or 2D dataset.")

        match scheme:
            case "NEQR":
                method = getattr(self, f"prepare_neqr_{suffix}")
            case "FRQI":
                method = getattr(self, f"prepare_frqi_{suffix}")
            case "AMPL":
                method = getattr(self, f"prepare_amplitude_{suffix}")
            case _:
                raise ValueError("Select one of NEQR, FRQI, or AMPL.")    

        target_state = method(data)
        print("\nPrepared target state = ")
        display(target_state.draw('latex'))

        print("\nTarget norm:", np.linalg.norm(target_state.data))
        print("Nonzero amplitudes:", np.count_nonzero(target_state.data))

        # train and minimize cost function
        
        parameters, circuit, best_fidelity = self.train_circuit(target_state, depth, epochs, learning_rate, loss_function_type)

        display(circuit.draw("mpl"))

        print("\nData = ", data)
        compare_similarlity_bw_two_states(target_state, circuit, self.n_index_q, self.n_feature_q, self.n_value_q)
        state_analysis(Statevector.from_instruction(circuit), "Encoded state", self.n_value_q, self.n_feature_q, self.n_index_q)

        # return the parameters
        return parameters, best_fidelity