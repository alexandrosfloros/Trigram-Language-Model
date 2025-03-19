import argparse
import os

import numpy as np


def read_model(file: str) -> dict[str, float]:
    with open(file) as f:
        model = {}

        for line in f:
            key, val = line.split("\t")
            model[key] = float(val)

        return model


def get_distribution(
    model_probs: dict[str, float], trigram_prefix: str
) -> tuple[list[str], list[float]]:
    outcomes = [
        outcome for outcome in model_probs if outcome.startswith(trigram_prefix)
    ]

    probs = [model_probs[outcome] for outcome in outcomes]
    probs_sum = sum(probs)
    probs = [probs[j] / probs_sum for j in range(len(probs))]

    return outcomes, probs


def generate_seqs(model_probs: dict[str, float], char_count: int) -> list[str]:
    initial_trigram_prefix = "##"

    (
        initial_trigram_outcomes,
        initial_trigram_probs,
    ) = get_distribution(model_probs, initial_trigram_prefix)

    trigram = np.random.choice(initial_trigram_outcomes, 1, p=initial_trigram_probs)[0]
    char = trigram[-1]
    seqs = [char]

    char_counter = 1

    while char_counter < char_count:
        if char != "#":
            next_trigram_prefix = trigram[-2:]

            (
                next_trigram_outcomes,
                next_trigram_probs,
            ) = get_distribution(model_probs, next_trigram_prefix)

        else:
            next_trigram_prefix = "##"

            (
                next_trigram_outcomes,
                next_trigram_probs,
            ) = get_distribution(model_probs, next_trigram_prefix)

        trigram = np.random.choice(next_trigram_outcomes, 1, p=next_trigram_probs)[0]
        char = trigram[-1]

        if char != "#":
            char_counter += 1

        seqs.append(char)

    seqs = [char for char in seqs if char != "#"]

    return seqs


def save_seqs(generated_seqs: list[str]) -> None:
    generated_seqs_file = "./results/generated_seqs.txt"

    with open(generated_seqs_file, "w") as f:
        entry = "".join(generated_seqs).replace(".", ".\n")
        f.write(entry)

    print(entry)


def main(model_file: str, char_count: int) -> None:
    os.mkdirs("./results", exist_ok=True)

    model_probs = read_model(model_file)
    generated_seqs = generate_seqs(model_probs, char_count)

    save_seqs(generated_seqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--chars", type=int, required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError("Model file is invalid.")

    if not "." in args.model:
        raise ValueError("Model file must have a language extension.")

    if args.chars <= 0:
        raise ValueError("Number of characters must be greater than 0.")

    main(args.model, args.chars)
