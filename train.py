import argparse
import itertools
import os
from random import shuffle
import re

import numpy as np
import matplotlib.pyplot as plt


def init_trigram_counts(trigram_chars: str) -> dict[str, int]:
    trigram_counts = {}

    for chars in itertools.product(trigram_chars, repeat=3):
        trigram = "".join(chars)

        if trigram[0] == trigram[2] == "#":
            continue

        if trigram[0] != "#" and trigram[1] == "#":
            continue

        trigram_counts[trigram] = 0

    return trigram_counts


def init_bigram_counts(trigram_chars: str) -> dict[str, int]:
    bigram_counts = {}

    for chars in itertools.product(trigram_chars, repeat=2):
        bigram = "".join(chars)

        if bigram[0] != "#" and bigram[1] == "#":
            continue

        bigram_counts[bigram] = 0

    return bigram_counts


def preprocess_line(line: str, regex) -> str:
    line = regex.sub("", line)
    line = re.sub(r"\d", "0", line)
    line = line.lower()

    return line


def get_file_lines(file: str) -> list[str]:
    with open(file) as f:
        return f.readlines()


def get_trigram_data(
    lines: list[str],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    unwanted_chars = re.compile(r"[^a-zA-Z0-9. ]")
    trigram_chars = " #.0abcdefghijklmnopqrstuvwxyz"

    trigram_counts = init_trigram_counts(trigram_chars)
    bigram_counts = init_bigram_counts(trigram_chars)

    trigrams = []

    for line in lines:
        line = preprocess_line(line, unwanted_chars)
        line_length = len(line)

        if line_length < 3:
            continue

        for i in range(line_length + 1):
            if i == 0:
                trigram = f"##{line[i]}"

            elif i == 1:
                trigram = f"#{line[i - 1: i + 1]}"

            elif i == line_length:
                trigram = f"{line[i - 2: i]}#"

            else:
                trigram = line[i - 2 : i + 1]

            trigram_counts[trigram] += 1

            bigram = trigram[:2]
            bigram_counts[bigram] += 1

            trigrams.append(trigram)

    return trigram_counts, bigram_counts, trigrams


def smooth_with_alpha(
    trigram_counts: dict[str, int], bigram_counts: dict[str, int], alpha: float
) -> dict[str, float]:
    trigram_vocabulary_size = 30
    trigram_probs = {}

    for trigram in trigram_counts:
        bigram = trigram[:-1]

        trigram_probs[trigram] = (trigram_counts[trigram] + alpha) / (
            bigram_counts[bigram] + alpha * trigram_vocabulary_size
        )

    return trigram_probs


def calculate_perplexity(
    training_trigram_probs: dict[str, float], trigrams: list[str]
) -> float:
    cross_entropy = 0

    for trigram in trigrams:
        probability = training_trigram_probs[trigram]
        cross_entropy -= np.log2(probability)

    cross_entropy /= len(trigrams)
    perplexity = 2**cross_entropy

    return perplexity


def get_validation_results(
    training_trigram_counts: dict[str, int],
    training_bigram_counts: dict[str, int],
    validation_trigrams: list[str],
    alphas: np.ndarray[int],
) -> tuple[dict[str, float], float, float, list[float]]:
    optimal_perplexity = float("inf")
    optimal_alpha = 0.0

    optimal_trigram_probs = {}
    validation_perplexities = []

    for alpha in alphas:
        training_trigram_probs = smooth_with_alpha(
            training_trigram_counts, training_bigram_counts, alpha
        )

        validation_perplexity = calculate_perplexity(
            training_trigram_probs, validation_trigrams
        )

        validation_perplexities.append(validation_perplexity)

        if validation_perplexity < optimal_perplexity:
            optimal_trigram_probs = training_trigram_probs
            optimal_perplexity = validation_perplexity
            optimal_alpha = alpha

    return (
        optimal_trigram_probs,
        optimal_alpha,
        optimal_perplexity,
        validation_perplexities,
    )


def get_testing_results(
    training_trigram_counts: dict[str, int],
    training_bigram_counts: dict[str, int],
    testing_trigrams: list[str],
    alphas: np.ndarray[int],
    optimal_alpha: float,
) -> tuple[list[float]]:
    testing_perplexities = []

    for alpha in alphas:
        training_trigram_probs = smooth_with_alpha(
            training_trigram_counts, training_bigram_counts, alpha
        )
        testing_perplexity = calculate_perplexity(
            training_trigram_probs, testing_trigrams
        )

        testing_perplexities.append(testing_perplexity)

        if alpha == optimal_alpha:
            model_perplexity = testing_perplexity

    return model_perplexity, testing_perplexities


def save_model(probs: dict[str, float], language: str):
    model_file = f"./results/model.{language}"

    with open(model_file, "w") as f:
        for trigram, prob in probs.items():
            exp_prob = format(prob, "e")

            f.write(f"{trigram}\t{exp_prob}\n")


def save_validation_perplexity(
    optimal_perplexity: float,
    optimal_alpha: float,
    language: str,
):
    validation_perplexity_file = f"./results/validation_perplexity_{language}.dat"

    with open(validation_perplexity_file, "w") as f:
        entry = f"Validation: perplexity={optimal_perplexity}, alpha={optimal_alpha}"
        f.write(entry)

    print(entry)


def save_testing_perplexity(model_perplexity: float, language: str):
    testing_perplexity_file = f"./results/testing_perplexity_{language}.dat"

    with open(testing_perplexity_file, "w") as f:
        entry = f"Testing: perplexity={model_perplexity}"
        f.write(entry)

    print(entry)


def save_perplexity_plot(
    alphas: np.ndarray[int],
    validation_perplexities: list[float],
    testing_perplexities: list[float],
    language: str,
):
    model_perplexity_plot_file = f"./results/perplexity_{language}.pdf"

    plt.plot(alphas, validation_perplexities)
    plt.plot(alphas, testing_perplexities)
    plt.xlim(alphas[0], alphas[-1])
    plt.xscale("log")
    plt.xlabel("Parameter α")
    plt.ylabel("Model Perplexity")
    plt.legend(["Validation Set", "Testing Set"])
    plt.title("Model Perplexity Using Validation and Testing Sets")
    plt.grid()

    plt.savefig(model_perplexity_plot_file, format="pdf")


def main(training_file: str, testing_file: str | None):
    if not os.path.exists("./results"):
        os.mkdir("./results")

    testing_file_provided = isinstance(testing_file, str)
    language = training_file.split(".")[-1]

    training_file_lines = get_file_lines(training_file)
    shuffle(training_file_lines)

    training_data = training_file_lines[:900]
    validation_data = training_file_lines[900:]

    training_trigram_counts, training_bigram_counts, _ = get_trigram_data(training_data)
    _, _, validation_trigrams = get_trigram_data(validation_data)

    alpha_count = 100
    min_alpha = 10**-4
    max_alpha = 1.0

    alphas = np.linspace(min_alpha, max_alpha, alpha_count)

    (
        optimal_trigram_probs,
        optimal_alpha,
        optimal_perplexity,
        validation_perplexities,
    ) = get_validation_results(
        training_trigram_counts,
        training_bigram_counts,
        validation_trigrams,
        alphas,
    )

    save_model(optimal_trigram_probs, language)
    save_validation_perplexity(
        optimal_perplexity,
        optimal_alpha,
        language,
    )

    if testing_file_provided:
        testing_data = get_file_lines(testing_file)
        _, _, testing_trigrams = get_trigram_data(testing_data)
        testing_perplexity, testing_perplexities = get_testing_results(
            training_trigram_counts,
            training_bigram_counts,
            testing_trigrams,
            alphas,
            optimal_alpha,
        )

        save_testing_perplexity(testing_perplexity, language)
        save_perplexity_plot(
            alphas,
            validation_perplexities,
            testing_perplexities,
            language,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training", type=str, required=True)
    parser.add_argument("--testing", type=str)

    args = parser.parse_args()

    if not os.path.isfile(args.training):
        raise FileNotFoundError("Training file is invalid.")

    if not "." in args.training:
        raise ValueError("Training file must have a language extension.")

    if (args.testing is not None) and (not os.path.isfile(args.testing)):
        raise FileNotFoundError("Testing file is invalid.")

    main(args.training, args.testing)
