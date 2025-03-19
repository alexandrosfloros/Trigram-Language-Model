# Trigram Language Model

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

## Language Identification

Once a trigram language model has been trained, its preplexity can be evaluated using a test document. If this text is applied on a set of models trained on different languages, the model matching the document's language should give the lowest perplexity, hence providing a method of performing language identification.
