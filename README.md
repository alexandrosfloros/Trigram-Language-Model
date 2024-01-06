# Trigram Language Model

Training and testing sets can be found in the data folder. These consist of sentences from the European Parlialiament Corpus in English (en), German (de) and Spanish (es). Once a trigram language model is trained, it can be tested using a separate document to evaluate its perplexity. If the same testing document is used on a set of models trained on different languages, the model matching the document's language should give the lowest perplexity, hence providing a method of performing language identification.

## Model Training

Train a trigram language model (saves model and validation perplexity):

```
python train.py --training <training_file.lang>
```

Train and simultaneously test a trigram language model (saves model, validation perplexity, testing perplexity and perplexity plot for validation and testing):

```
python train.py --training <training_file.lang> --testing <testing_file>
```

## Model Testing

Test a trigram language model (saves testing perplexity):

```
python test.py --model <model_file.lang> --testing <testing_file>
```

## Sequence Generation

Generate sequences of characters from a trigram language model (saves generated sequences):

```
python generate.py --model <model_file.lang> --chars <character_count>
```