import argparse
import os

from generate import *
from train import *


def main(model_file: str, testing_file: str) -> None:
    # make results directory
    if not os.path.exists("./results"):
        os.mkdir("./results")

    # get language
    language = model_file.split(".")[-1]

    # get model dictionary
    model_probs = read_model(model_file)

    # get testing file lines
    testing_file_lines = get_file_lines(testing_file)

    # get testing trigram list
    _, _, testing_trigrams = get_trigram_data(testing_file_lines)

    # get perplexity
    testing_perplexity = calculate_perplexity(model_probs, testing_trigrams)

    # save testing perplexity
    save_testing_perplexity(testing_perplexity, language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--testing", type=str, required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError("Model file is invalid.")

    if not "." in args.model:
        raise ValueError("Model file must have a language extension.")

    if not os.path.isfile(args.testing):
        raise FileNotFoundError("Testing file is invalid.")

    main(args.model, args.testing)
