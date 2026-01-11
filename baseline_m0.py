import numpy as np
from collections import Counter
from torchvision.datasets import SVHN

def main():
    train_ds = SVHN(root="./data", split="train", download=True)
    test_ds  = SVHN(root="./data", split="test", download=True)

    train_labels = train_ds.labels
    test_labels = test_ds.labels

    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)

    print("=== Distribution TRAIN ===")
    for k in sorted(train_dist):
        print(f"Classe {k}: {train_dist[k]}")

    print("\n=== Distribution TEST ===")
    for k in sorted(test_dist):
        print(f"Classe {k}: {test_dist[k]}")

    majority_class = train_dist.most_common(1)[0][0]
    print("\nClasse majoritaire (train):", majority_class)

    majority_acc = np.mean(test_labels == majority_class)
    print("Accuracy classe majoritaire (test):", majority_acc)

    rng = np.random.default_rng(seed=42)
    random_preds = rng.integers(low=0, high=10, size=len(test_labels))
    random_acc = np.mean(random_preds == test_labels)
    print("Accuracy aléatoire uniforme (test):", random_acc)

    print("\nAccuracy hasard théorique =", 1/10)

if __name__ == "__main__":
    main()
