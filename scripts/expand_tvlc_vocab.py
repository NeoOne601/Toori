import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

def _sanitize_text(value: str) -> str:
    import re
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", str(value or "").lower())
    return " ".join(cleaned.split()).strip()

def main():
    weights_path = "models/vision/tvlc_connector.npz"
    cache_dir = Path(".toori/tvlc_train_cache")
    output_path = "models/vision/tvlc_connector.npz"
    max_prototypes = 4096

    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return

    print(f"Loading existing weights from {weights_path}...")
    data = dict(np.load(weights_path, allow_pickle=True))
    
    print(f"Scanning cache directory for Gemma targets: {cache_dir}")
    # We look for the _gemma.npz files created by the precacher
    gemma_files = list(cache_dir.glob("*_gemma.npz"))
    
    if not gemma_files:
        print("No _gemma.npz cache files found. Expansion impossible.")
        return

    print(f"Found {len(gemma_files)} semantic targets.")
    
    accum = defaultdict(list)
    label_counts = Counter()

    print("Extracting high-fidelity semantic prototypes...")
    # Process up to 100,000 files to maximize vocabulary capture
    for f in tqdm(gemma_files[:100000]): 
        try:
            d = np.load(str(f), allow_pickle=True)
            if "primary_label" not in d or "target_tokens" not in f.name:
                # Some versions might save keys differently, let's look for tokens
                pass
            
            label = _sanitize_text(str(d["primary_label"]))
            if not label or label in ["object", "none", "unknown", ""]:
                continue
            
            # target_tokens in precache are [32, 2048]
            # We take the mean across the 32 slots as the central semantic vector for this label
            target_tokens = d["target_tokens"]
            central_vector = target_tokens.mean(axis=0)
            central_vector /= np.linalg.norm(central_vector) + 1e-9
            
            accum[label].append(central_vector)
            label_counts[label] += 1
        except Exception:
            continue

    print(f"Collected {len(accum)} unique labels from cache.")
    # Pick the top N most frequent labels to keep the runtime hot-path efficient
    top_labels = [l for l, c in label_counts.most_common(max_prototypes)]
    
    prototype_labels = []
    prototype_vectors = []
    
    for label in top_labels:
        vectors = accum[label]
        # Final prototype is the average of all sightings of this label in COCO
        final_vec = np.stack(vectors).mean(axis=0)
        final_vec /= np.linalg.norm(final_vec) + 1e-9
        prototype_labels.append(label)
        prototype_vectors.append(final_vec)

    if not prototype_labels:
        print("Failed to extract any valid prototypes.")
        return

    # Update the weight file
    data["prototype_labels"] = np.array(prototype_labels)
    data["prototype_vectors"] = np.stack(prototype_vectors).astype(np.float32)
    
    print(f"Success! Expanded vocabulary to {len(prototype_labels)} semantic prototypes.")
    print(f"Saving updated weights to {output_path}...")
    np.savez(output_path, **data)
    print("Optimization complete. Runtime will now recognize a much wider range of objects.")

if __name__ == "__main__":
    main()
