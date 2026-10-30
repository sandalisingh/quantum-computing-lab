from pathlib import Path
import json
import numpy as np


def save_parameters(
    parameters, depth, epochs, learning_rate, fidelity,
    scheme="NEQR", loss_function_type="FIDELITY"
):
    """Save trained dataset encoder parameters."""

    metadata = {
        "scheme": scheme,
        "depth": int(depth),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss_function_type": loss_function_type,
        "fidelity": float(fidelity)
    }

    filename = Path("image_matching_" + scheme + "_" + loss_function_type + ".npz")

    np.savez(
        filename,
        parameters=np.asarray(parameters),
        metadata=np.asarray(json.dumps(metadata))
    )

    print(f"Dataset encoder saved to: {filename}")


def load_parameters(
    scheme="NEQR",
    loss_function_type="FIDELITY"
):
    """Load and return the complete saved encoder information."""

    filename = Path("image_matching_" + scheme + "_" + loss_function_type + ".npz")

    if not filename.exists():
        raise FileNotFoundError(f"No saved encoding found at: {filename}")

    with np.load(filename, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        parameters = saved["parameters"].copy()

    # Validate encoder settings
    if (
        metadata["scheme"] != scheme
        or metadata["loss_function_type"] != loss_function_type
    ):
        raise ValueError(
            "Saved encoding does not match the current settings."
        )

    # Return the complete saved object
    return {
        "parameters": parameters,
        "fidelity": metadata["fidelity"],
        "depth": metadata["depth"],
        "epochs": metadata["epochs"],
        "learning_rate": metadata["learning_rate"],
        "scheme": metadata["scheme"],
        "loss_function_type": metadata["loss_function_type"]
    }
