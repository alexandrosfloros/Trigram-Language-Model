import argparse
import itertools
import os
from random import shuffle
import re

import numpy as np
import matplotlib.pyplot as plt


def init_trigram_counts(trigram_chars: str) -> dict[str, int]:
    # initialise trigram count dictionary
    trigram_counts = {}

    # set trigram counts to zero
    for chars in itertools.product(trigram_chars, repeat=3):
        trigram = "".join(chars)

        # trigrams with # can only be of form #xx, ##x or xx#
        if trigram[0] == trigram[2] == "#":
            continue
        if trigram[0] != "#" and trigram[1] == "#":
            continue

        trigram_counts[trigram] = 0

    return trigram_counts


def init_bigram_counts(trigram_chars: str) -> dict[str, int]:
    # initialise bigram count dictionary
    bigram_counts = {}

    # set bigram counts to zero
    for chars in itertools.product(trigram_chars, repeat=2):
        bigram = "".join(chars)

        # bigrams with # can only be of form #x or ##
        if bigram[0] != "#" and bigram[1] == "#":
            continue

        bigram_counts[bigram] = 0

    return bigram_counts


def preprocess_line(line: str, regex) -> str:
    # remove unwanted characters
    line = regex.sub("", line)

    # replace digits with 0
    line = re.sub(r"\d", "0", line)

    # convert to lowercase
    line = line.lower()

    return line


def get_file_lines(file: str) -> list[str]:
    with open(file) as f:
        return f.readlines()


def get_trigram_data(
    lines: list[str],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    # get unwanted characters
    unwanted_chars = re.compile(r"[^a-zA-Z0-9. ]")

    # get trigram characters
    trigram_chars = " #.0abcdefghijklmnopqrstuvwxyz"

    # initialise trigram counts
    trigram_counts = init_trigram_counts(trigram_chars)

    # initialise bigram counts
    bigram_counts = init_bigram_counts(trigram_chars)

    # initialise trigrams list
    trigrams = []

    for line in lines:
        # preprocess line
        line = preprocess_line(line, unwanted_chars)

        # get line length
        line_length = len(line)

        # skip lines with no trigrams
        if line_length < 3:
            continue

        # get trigrams
        for i in range(line_length + 1):
            # first character follows "start of sequence" (##)
            if i == 0:
                trigram = f"##{line[i]}"

            # second character
            elif i == 1:
                trigram = f"#{line[i - 1: i + 1]}"

            # after last character
            # last trigram ends with "end of sequence" (#)
            elif i == line_length:
                trigram = f"{line[i - 2: i]}#"

            # other characters
            else:
                trigram = line[i - 2 : i + 1]

            # increase trigram count
            trigram_counts[trigram] += 1

            # increase bigram count
            bigram = trigram[:2]
            bigram_counts[bigram] += 1

            # add trigram to trigrams
            trigrams.append(trigram)

    return trigram_counts, bigram_counts, trigrams


def smooth_with_alpha(
    trigram_counts: dict[str, int], bigram_counts: dict[str, int], alpha: float
) -> dict[str, float]:
    # set trigram vocabulary size
    trigram_vocabulary_size = 30

    # initialise trigram probabilities
    trigram_probs = {}

    # return alpha smoothed model distribution
    for trigram in trigram_counts:
        # set bigram
        bigram = trigram[:-1]

        # set trigram probabilities
        trigram_probs[trigram] = (trigram_counts[trigram] + alpha) / (
            bigram_counts[bigram] + alpha * trigram_vocabulary_size
        )

    return trigram_probs


def calculate_perplexity(
    training_trigram_probs: dict[str, float], trigrams: list[str]
) -> float:
    # initialise cross entropy
    cross_entropy = 0

    # calculate cross entropy
    for trigram in trigrams:
        # get trigram probability
        probability = training_trigram_probs[trigram]

        # update cross entropy
        cross_entropy -= np.log2(probability)

    # normalise cross entropy
    cross_entropy /= len(trigrams)

    # calculate perplexity
    perplexity = 2**cross_entropy

    return perplexity


def get_validation_results(
    training_trigram_counts: dict[str, int],
    training_bigram_counts: dict[str, int],
    validation_trigrams: list[str],
    alphas: np.ndarray[int],
) -> tuple[dict[str, float], float, float, list[float]]:
    # initialise optimal trigram probabilities
    optimal_trigram_probs = {}

    # initialise optimal perplexity
    optimal_perplexity = float("inf")

    # initialise optimal alpha
    optimal_alpha = 0.0

    # initialise validation perplexities
    validation_perplexities = []

    # get optimal perplexity
    for alpha in alphas:
        # get training trigram probabilities with smoothing
        training_trigram_probs = smooth_with_alpha(
            training_trigram_counts, training_bigram_counts, alpha
        )

        # get validation perplexity
        validation_perplexity = calculate_perplexity(
            training_trigram_probs, validation_trigrams
        )

        # add validation perplexity to validation perplexities
        validation_perplexities.append(validation_perplexity)

        # update optimal perplexity
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
    # initialise testing perplexities
    testing_perplexities = []

    for alpha in alphas:
        # get testing trigram probabilities with smoothing
        training_trigram_probs = smooth_with_alpha(
            training_trigram_counts, training_bigram_counts, alpha
        )
        # get testing perplexity
        testing_perplexity = calculate_perplexity(
            training_trigram_probs, testing_trigrams
        )

        # add testing perplexity to testing perplexities
        testing_perplexities.append(testing_perplexity)

        if alpha == optimal_alpha:
            model_perplexity = testing_perplexity

    return model_perplexity, testing_perplexities


def save_model(probs: dict[str, float], language: str):
    # set model file
    model_file = f"./results/model.{language}"

    with open(model_file, "w") as f:
        for trigram, prob in probs.items():
            # set probabilities to exponential format
            exp_prob = format(prob, "e")

            # save probabilities
            f.write(f"{trigram}\t{exp_prob}\n")


def save_validation_perplexity(
    optimal_perplexity: float,
    optimal_alpha: float,
    language: str,
):
    # set validation perplexity file
    validation_perplexity_file = f"./results/validation_perplexity_{language}.dat"

    # save validation perplexity and alpha
    with open(validation_perplexity_file, "w") as f:
        entry = f"Validation: perplexity={optimal_perplexity}, alpha={optimal_alpha}"
        f.write(entry)

    # print validation perplexity and alpha
    print(entry)


def save_testing_perplexity(model_perplexity: float, language: str):
    # set testing perplexity file
    testing_perplexity_file = f"./results/testing_perplexity_{language}.dat"

    # save testing perplexity
    with open(testing_perplexity_file, "w") as f:
        entry = f"Testing: perplexity={model_perplexity}"
        f.write(entry)

    # print testing perplexity
    print(entry)


def save_perplexity_plot(
    alphas: np.ndarray[int],
    validation_perplexities: list[float],
    testing_perplexities: list[float],
    language: str,
):
    # set perplexity plot file
    model_perplexity_plot_file = f"./results/perplexity_{language}.pdf"

    # plot perplexities
    plt.plot(alphas, validation_perplexities)
    plt.plot(alphas, testing_perplexities)
    plt.xlim(alphas[0], alphas[-1])
    plt.xscale("log")
    plt.xlabel("Parameter α")
    plt.ylabel("Model Perplexity")
    plt.legend(["Validation Set", "Testing Set"])
    plt.title("Model Perplexity Using Validation and Testing Sets")
    plt.grid()

    # save perplexity plot
    plt.savefig(model_perplexity_plot_file, format="pdf")


def main(training_file: str, testing_file: str | None):
    # make results directory
    if not os.path.exists("./results"):
        os.mkdir("./results")

    # whether testing file is provided
    testing_file_provided = isinstance(testing_file, str)

    # get language
    language = training_file.split(".")[-1]

    # get training file lines
    training_file_lines = get_file_lines(training_file)

    # shuffle training file lines
    shuffle(training_file_lines)

    # get training data
    training_data = training_file_lines[:900]

    # get validation data
    validation_data = training_file_lines[900:]

    # get train trigram and bigram counts
    training_trigram_counts, training_bigram_counts, _ = get_trigram_data(training_data)

    # get validation trigram list
    _, _, validation_trigrams = get_trigram_data(validation_data)

    # set alpha precision
    alpha_count = 100

    # set min alpha
    min_alpha = 10**-4

    # set max alpha
    max_alpha = 1.0

    # get alpha list
    alphas = np.linspace(min_alpha, max_alpha, alpha_count)

    # get validation results
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

    # save model
    save_model(optimal_trigram_probs, language)

    # save validation perplexity
    save_validation_perplexity(
        optimal_perplexity,
        optimal_alpha,
        language,
    )

    if testing_file_provided:
        # get testing data
        testing_data = get_file_lines(testing_file)

        # get testing trigram list
        _, _, testing_trigrams = get_trigram_data(testing_data)

        # get testing results
        testing_perplexity, testing_perplexities = get_testing_results(
            training_trigram_counts,
            training_bigram_counts,
            testing_trigrams,
            alphas,
            optimal_alpha,
        )

        # save testing perplexity
        save_testing_perplexity(testing_perplexity, language)

        # save perplexity plot
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
