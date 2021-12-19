import spacy
from spacy.tokens import DocBin
from collections import namedtuple
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import restore_data, categories
import matplotlib.pyplot as plt

Prediction = namedtuple('Prediction', ['cat', 'score'])


def extract_predicted_cat(doc):
    pred = sorted(doc.cats.items(), key=lambda item: item[1], reverse=True)[0]
    return Prediction(cat=pred[0], score=pred[1])


def predict(data, nlp):
    """
    Given a list of documents and a trained model, predicts category of each document
    :param data: List of SpaCy Docs
    :param nlp: SpaCy pipeline with a traind TextCatagorizer component
    :return: List of SpaCy Docs, each with an attribute 'cat' holding the predicted category
    """
    predictions = []
    print(f"processing {len(data)} documents. that might take a while...")
    for idx, doc in enumerate(nlp.pipe(data)):
        # scores = sorted(doc.cats.items(), key=lambda item: item[1], reverse=True)
        predictions.append(extract_predicted_cat(doc))

    return predictions


def get_true_label(doc):
    """
    Extracts from the true (as opposed to predicted) label of a SpaCy Doc
    :param doc: SpaCy Doc
    :return: label; Str
    """
    return list(filter(lambda elem: elem[1], doc.cats.items()))[0][0]


def results_as_df(labels, predictions):
    """
    Packs prediction results as a DataFrame
    :param labels: List of true categories
    :param predictions: List of Prediction tuples
    :return: Pandas DataFrame with: true category, predicted category, score of predicted category
    """
    return pd.DataFrame({"label": labels,
                         "predicted": [p.cat for p in predictions],
                         "score": [p.score for p in predictions]})


def evaluate(labels, predictions, categories):
    """
    Calculates confusion matrix and overall accuracy
    :param labels: List of true categories
    :param predictions: List of Prediction tuples
    :param categories: List of unique categories in the data
    :return: accuracy (TPR) and an sklearn confusion matrix
    """
    class_status = [label == prediction.cat for label, prediction in zip(labels, predictions)]
    accuracy = sum(class_status) / len(class_status)
    print(f"overall accuracy: {accuracy:.2f}")
    conf_mat = confusion_matrix(y_true=labels, y_pred=[p.cat for p in predictions], labels=categories)
    return accuracy, conf_mat


if __name__ == '__main__':
    doc_classfier = spacy.load("./output/model-best")
    test = restore_data("./data/test.spacy", doc_classfier)
    labels = [get_true_label(doc) for doc in test]
    predictions = predict(test, doc_classfier)

    # save test set with model predictions as DocBin, to easily fetch results for analysis
    test_doc_bin = DocBin(docs=test)
    test_doc_bin.to_disk("./data/test_with_pred.spacy")

    # results can be evaluated at this stage
    results = results_as_df(labels=labels, predictions=predictions)
    acc, conf = evaluate(labels=labels, predictions=predictions, categories=categories)

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=categories)
    disp.plot(xticks_rotation='vertical')




