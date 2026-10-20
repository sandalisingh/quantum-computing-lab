import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_distribution

# TO GET INFO ABOUT A SINGLE QUANTUM STATE

def print_state_info_tabular(state, title, n_value_q, n_feature_q, n_index_q):
    print(f"\n{title}")
    rows = extract_fields_from_q_state(state, n_value_q, n_feature_q, n_index_q)

    is_query = state.num_qubits == n_value_q + n_feature_q
    n_values = 2 ** n_value_q
    rows_2 = {}

    for row, feature, val, amp in rows:
        key = (feature,) if is_query else (row, feature)
        rows_2.setdefault(key, [0j] * n_values)[val] = amp

    headers = ([] if is_query else ["Row"]) + ["Feature"] + [
        f"Value-{v}-Amplitude" for v in range(n_values)
    ]
    width = 25

    print("".join(f"{h:<{width}}" for h in headers))
    print("-" * width * len(headers))

    for key, amps in sorted(rows_2.items()):
        data = list(map(str, key)) + [
            f"{a.real:+.3f}" + (f" {a.imag:+.3f}j" if abs(a.imag) > 1e-10 else "")
            for a in amps
        ]
        print("".join(f"{item:<{width}}" for item in data))

    return rows_2
        
def extract_fields_from_q_state(state, n_value_q, n_feature_q, n_index_q):
    '''Defined for NEQR encoding scheme
    For dataset:
    |index> |feature no> |value>
    For query:
    |feature no> |value>
    '''

    total_qubits = state.num_qubits
    query_qubits = n_value_q + n_feature_q

    is_query = (total_qubits == query_qubits)

    extracted_dict = []

    # STRING SLICING : string[start : stop : step]

    for basis, amplitude in enumerate(state.data):
        if abs(amplitude) > 1e-4:
            basis_string = format(basis, f'0{total_qubits}b')

            if is_query:
                # Query format: feature (n_feature_q) + value (n_value_q)
                index = 0
                
                feature_string = basis_string[:n_feature_q]
                feature = int(feature_string, 2)
                
                value_string = basis_string[n_feature_q:]
                value = int(value_string, 2) if value_string else 0
            else:
                # Dataset format: index (n_index_q) + feature (n_feature_q) + value (n_value_q)
                # Read from left to right:
                idx_end = n_index_q
                feat_end = n_index_q + n_feature_q
                
                index_string = basis_string[:idx_end]
                index = int(index_string, 2)
                
                feature_string = basis_string[idx_end:feat_end]
                feature = int(feature_string, 2)
                
                value_string = basis_string[feat_end:]
                value = int(value_string, 2) if value_string else 0

            extracted_dict.append([index, feature, value, amplitude])

    return extracted_dict

def plot_value_probabilities(state, title, n_value_q, n_feature_q, n_index_q):
    rows = extract_fields_from_q_state(state, n_value_q, n_feature_q, n_index_q)
    is_query = (state.num_qubits == n_value_q + n_feature_q)
    n_values = 2 ** n_value_q
    pairs = sorted(set((i, f) for i, f, v, a in rows))

    probs = np.zeros((n_values, len(pairs)))
    positions = {pair: x for x, pair in enumerate(pairs)}

    for i, f, v, a in rows:
        probs[v, positions[(i, f)]] = abs(a) ** 2

    labels = [f"F{f}" if is_query else f"I{i},F{f}" for i, f in pairs]
    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.6), 5))

    bottom = np.zeros(len(pairs))
    for v, p in enumerate(probs):
        ax.bar(labels, p, bottom=bottom, label=f"Value {v}")
        bottom += p

    ax.set(title=title, xlabel="Index, Feature" if not is_query else "Feature",
           ylabel="Probability")
    ax.tick_params(axis="x", labelrotation=90 if len(pairs) > 10 else 0)
    ax.legend(title="Value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

def plot_signed_amplitudes(state, title, n_value_q, n_feature_q, n_index_q):
    rows = extract_fields_from_q_state(state, n_value_q, n_feature_q, n_index_q)
    is_query = (state.num_qubits == n_value_q + n_feature_q)

    labels = []
    amplitudes = []

    for i, f, v, a in rows:
        if abs(a) > 1e-5:
            if is_query:
                label = f"F{f}, V{v}"
            else:
                label = f"I{i}, F{f}, V{v}"

            labels.append(label)
            amplitudes.append(np.real(a))

    fig, ax = plt.subplots(figsize=(25,6))
    ax.bar(labels, amplitudes)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set(
        title=title,
        xlabel="Index, Feature, Value" if not is_query else "Feature, Value",
        ylabel="Signed Amplitude"
    )
    ax.tick_params(axis="x", labelrotation=90 if len(labels) > 10 else 0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

def state_analysis(state, title, n_value_q, n_feature_q, n_index_q):
    '''Defined for NEQR encoding scheme
    For dataset:
    |value> |feature no> |index>
    For query:
    |value> |feature no>
    '''
    print("\nState vector:")
    display(state.draw('latex'))

    display_basis_and_amplitudes(state)    
    plot_signed_amplitudes(state, title, n_value_q, n_feature_q, n_index_q)
    plot_basis_vs_probs(state, title)

    plot_value_probabilities(state, title, n_value_q, n_feature_q, n_index_q)
    print_state_info_tabular(state, title, n_value_q, n_feature_q, n_index_q)
    display_index_wise_probabilities(state, title, n_value_q, n_feature_q, n_index_q)
    
def plot_basis_vs_probs(state, title):
    state_probs = state.probabilities_dict()

    # Check whether the sum of probabilities equals 1
    total_prob = sum(state_probs.values())
    print(f"\nSum of probabilities: {total_prob:.10f}")

    if np.isclose(total_prob, 1.0):
        print("Check passed: Sum of probabilities equals 1.")
    else:
        print("Warning: Sum of probabilities does not equal 1!")

    fig = plot_distribution(
        state_probs, 
        title=title,
        bar_labels=True 
    )
    ax = fig.axes[0]
    for text in ax.texts:
        text.set_rotation(90)
        text.set_ha('center')    # Center the text over the bar
        text.set_va('bottom')    # Keep text cleanly above the bar
    fig.set_size_inches(25, 6)
    fig.tight_layout()
    display(fig)

def display_basis_and_amplitudes(state):
    amplitudes = state.data
    total_qubits = state.num_qubits
    print("\nBasis states and amplitudes:")

    for i, amp in enumerate(amplitudes):
        if abs(amp) > 1e-5:
            basis = format(i, f"0{total_qubits}b")
            print(f"|{basis}> : {amp}")

# TO COMPARE THE SIMILARLITY OF TWO QUANTUM STATES

def compare_similarlity_bw_two_states(target_state, circuit, n_index_q, n_feature_q, n_value_q):
    print("\nTarget state:")
    display(target_state.draw('latex'))
    encoded_state = Statevector.from_instruction(circuit)
    print("\nEncoded state: ")
    display(encoded_state.draw('latex'))

    fidelity = np.abs(np.vdot(target_state.data, encoded_state.data)) ** 2
    print(f"\nFidelity between target and encoded state: {fidelity:.8f}")
    
    heatmap_to_compare_states(target_state, encoded_state)
    basis_vs_prob_plot_to_compare_states(target_state, encoded_state)

    print_state_info_tabular(
        target_state, "TARGET STATE",
        n_value_q, n_feature_q, n_index_q
    )

    print_state_info_tabular(
        encoded_state, "ENCODED STATE",
        n_value_q, n_feature_q, n_index_q
    )

def heatmap_to_compare_states(target_state, encoded_state):
    # Get probabilities for both states
    target_probs = target_state.probabilities()
    encoded_probs = encoded_state.probabilities()

    # Heatmap: Target vs Encoded probabilities
    comparison = np.vstack([target_probs, encoded_probs])

    plt.figure(figsize=(16, 3))
    fig = sns.heatmap(
        comparison,
        cmap="viridis",
        xticklabels=False,
        yticklabels=["Target State", "Encoded PQC State"],
        cbar_kws={"label": "Probability"}
    )
    plt.xlabel("State index")
    plt.title(f"Target vs Encoded State")
    plt.tight_layout()
    plt.show()
    display(fig)

def basis_vs_prob_plot_to_compare_states(target_state, encoded_state):
    target_probs = target_state.probabilities_dict()
    encoded_probs = encoded_state.probabilities_dict()
    fig = plot_distribution(
        [target_probs, encoded_probs], 
        legend=["Target State", "Encoded PQC State"],
        color=["#1f77b4", "#ff7f0e"], # Distinct blue and orange bars
        title=f"Quantum State Overlay Comparison",
        bar_labels=True 
    )
    ax = fig.axes[0]
    for text in ax.texts:
        text.set_rotation(90)
        text.set_ha('center')    # Center the text over the bar
        text.set_va('bottom')    # Keep text cleanly above the bar
    fig.set_size_inches(25, 6)
    fig.tight_layout()
    display(fig)

def display_index_wise_probabilities(state, title, n_value_q, n_feature_q, n_index_q):
    total_qubits = state.num_qubits
    probabilities = state.probabilities()

    # Check whether this is a dataset state or query state
    is_dataset = (total_qubits == n_value_q + n_feature_q + n_index_q)
    if not is_dataset:
        return

    N_I = 2 ** n_index_q
    index_probs = np.zeros(N_I)

    for basis_index, probability in enumerate(probabilities):
        # Index occupies the most significant bits
        row_index = basis_index >> (n_value_q + n_feature_q)
        index_probs[row_index] += probability

    # Plot index-wise probabilities
    indices = np.arange(N_I)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(indices, index_probs)

    plt.xlabel("Dataset Index (Row)")
    plt.ylabel("Probability")
    plt.title(f"Index-wise Probabilities - {title}")
    plt.xticks(indices)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Display probability above each bar
    for bar, probability in zip(bars, index_probs):
        if probability > 1e-8:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{probability:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()
