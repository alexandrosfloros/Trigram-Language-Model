import argparse
import os

import numpy as np


def read_model(file: str) -> dict[str, float]:
    with open(file) as f:
        model = {}

        # read columns and add to dictionary
        for line in f:
            key, val = line.split("\t")
            model[key] = float(val)

        return model


def get_distribution(
    model_probs: dict[str, float], trigram_prefix: str
) -> tuple[list[str], list[float]]:
    # get outcomes and probabilities of possible next trigrams starting with "trigram_prefix" prefix
    outcomes = [
        outcome for outcome in model_probs if outcome.startswith(trigram_prefix)
    ]
    probs = [model_probs[outcome] for outcome in outcomes]

    # normalise probabilities
    probs_sum = sum(probs)
    probs = [probs[j] / probs_sum for j in range(len(probs))]

    return outcomes, probs


def generate_seqs(model_probs: dict[str, float], char_count: int) -> list[str]:
    # set initial trigram prefix
    initial_trigram_prefix = "##"

    # get outcomes and probabilities of possible next trigrams starting with "initial_trigram_prefix" prefix
    (
        initial_trigram_outcomes,
        initial_trigram_probs,
    ) = get_distribution(model_probs, initial_trigram_prefix)

    # get first trigram from new distribution
    trigram = np.random.choice(initial_trigram_outcomes, 1, p=initial_trigram_probs)[0]

    # get last character from first trigram
    char = trigram[-1]

    # initialise sequence list
    seqs = [char]

    # initialise character counter
    char_counter = 1

    # generate rest of characters following first trigram
    while char_counter < char_count:
        # if next character does not follow "end of sequence" (#), it belongs to same sequence and has context
        if char != "#":
            # use last two characters of previous trigram ("trigram"), which are first two characters of next trigram, to predict next character (last character of next trigram)
            next_trigram_prefix = trigram[-2:]

            # get outcomes and probabilities of possible next trigrams starting with "next_trigram_prefix" prefix
            (
                next_trigram_outcomes,
                next_trigram_probs,
            ) = get_distribution(model_probs, next_trigram_prefix)

        # if next character follows "end of sequence" (#), it belongs to new sequence and has no context
        else:  # char == "#"
            # assume context of "start of sequence" (##)
            next_trigram_prefix = "##"

            # get outcomes and probabilities of possible next trigrams starting with "start of sequence" (##) prefix
            (
                next_trigram_outcomes,
                next_trigram_probs,
            ) = get_distribution(model_probs, next_trigram_prefix)

        # get next trigram from new distribution
        trigram = np.random.choice(next_trigram_outcomes, 1, p=next_trigram_probs)[0]

        # get last character from next trigram
        char = trigram[-1]

        # update character counter
        if char != "#":
            char_counter += 1

        # add next character to sequence list
        seqs.append(char)

    # remove hashes from sequence list
    seqs = [char for char in seqs if char != "#"]

    return seqs


def save_seqs(generated_seqs: list[str]) -> None:
    # set generated sequence file
    generated_seqs_file = "./results/generated_seqs.txt"

    # save generated sequences
    with open(generated_seqs_file, "w") as f:
        entry = "".join(generated_seqs).replace(".", ".\n")
        f.write(entry)

    # print sequences
    print(entry)


def main(model_file: str, char_count: int) -> None:
    # make results directory
    if not os.path.exists("./results"):
        os.mkdir("./results")

    # get model probabilities
    model_probs = read_model(model_file)

    # get generated sequences
    generated_seqs = generate_seqs(model_probs, char_count)

    # save generated sequences
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
