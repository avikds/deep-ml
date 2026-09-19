import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0

    counts = Counter(labels)
    total = len(labels)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def calculate_information_gain(
    examples: list[dict],
    attr: str,
    target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    if not examples:
        return 0.0

    # Entropy before splitting.
    parent_labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(parent_labels)

    total = len(examples)
    weighted_entropy = 0.0

    # Process values in sorted order for deterministic behavior.
    values = sorted({example[attr] for example in examples})

    for value in values:
        subset = [example for example in examples if example[attr] == value]

        if subset:
            labels = [example[target_attr] for example in subset]
            weight = len(subset) / total
            weighted_entropy += weight * calculate_entropy(labels)

    return parent_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(example[target_attr] for example in examples)

    if not counts:
        return None

    # Sort alphabetically first, then select the largest count.
    # Python's max preserves the first occurrence on ties.
    return max(sorted(counts), key=lambda label: counts[label])


def learn_decision_tree(
    examples: list[dict],
    attributes: list[str],
    target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    # No examples: no meaningful tree can be constructed.
    if not examples:
        return {}

    labels = [example[target_attr] for example in examples]

    # If all examples have the same class, return a leaf.
    if len(set(labels)) == 1:
        return labels[0]

    # No attributes remain: return the majority class.
    if not attributes:
        return majority_class(examples, target_attr)

    # Choose attribute with maximum information gain.
    # max() preserves the first attribute on equal gains,
    # satisfying the specified tie-breaking rule.
    best_attr = attributes[0]
    best_gain = calculate_information_gain(
        examples, best_attr, target_attr
    )

    for attr in attributes[1:]:
        gain = calculate_information_gain(
            examples, attr, target_attr
        )

        # Strictly greater preserves the earlier attribute on ties.
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}

    # Remove the selected attribute for recursive branches.
    remaining_attributes = [
        attr for attr in attributes if attr != best_attr
    ]

    # Process attribute values in sorted order.
    values = sorted({example[best_attr] for example in examples})

    for value in values:
        subset = [
            example
            for example in examples
            if example[best_attr] == value
        ]

        if not subset:
            tree[best_attr][value] = majority_class(
                examples, target_attr
            )
        else:
            tree[best_attr][value] = learn_decision_tree(
                subset,
                remaining_attributes,
                target_attr
            )

    return tree