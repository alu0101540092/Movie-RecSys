import pickle
import os
import numpy as np
from src.model import MODEL_PATH, BASE_DIR


def optimize():
    """
    Extracts and optimizes individual SVD model components for faster loading.

    Loads the full pickled SVD model, extracts the latent factor matrices (pu, qi)
    and bias vectors (bu, bi), and saves them as separate numpy arrays.
    It also saves the global mean and ID mappings.
    """
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, dict) and "algo" in data:
            algo = data["algo"]
        else:
            algo = data

    print("Model loaded. Extracting matrices...")

    models_dir = os.path.dirname(MODEL_PATH)

    # Extract and save components
    if hasattr(algo, "pu"):
        np.save(os.path.join(models_dir, "svd_pu.npy"), algo.pu)
        print("Saved svd_pu.npy")

    if hasattr(algo, "qi"):
        np.save(os.path.join(models_dir, "svd_qi.npy"), algo.qi)
        print("Saved svd_qi.npy")

    if hasattr(algo, "bu"):
        np.save(os.path.join(models_dir, "svd_bu.npy"), algo.bu)
        print("Saved svd_bu.npy")

    if hasattr(algo, "bi"):
        np.save(os.path.join(models_dir, "svd_bi.npy"), algo.bi)
        print("Saved svd_bi.npy")

    # Global mean
    if hasattr(algo, "trainset") and hasattr(algo.trainset, "global_mean"):
        np.save(
            os.path.join(models_dir, "svd_global_mean.npy"),
            np.array([algo.trainset.global_mean]),
        )
        print(f"Saved svd_global_mean.npy: {algo.trainset.global_mean}")
    elif hasattr(algo, "default_prediction"):  # Fallback
        np.save(
            os.path.join(models_dir, "svd_global_mean.npy"),
            np.array([algo.default_prediction]),
        )
        print(
            f"Saved svd_global_mean.npy (from default_prediction): {algo.default_prediction}"
        )

    # Save ID mappings
    if hasattr(algo, "trainset"):
        mappings = {
            "users": algo.trainset._raw2inner_id_users,
            "items": algo.trainset._raw2inner_id_items,
        }
        with open(os.path.join(models_dir, "svd_mappings.pkl"), "wb") as f:
            pickle.dump(mappings, f)
        print("Saved svd_mappings.pkl")

    print("Optimization complete.")


if __name__ == "__main__":
    optimize()
