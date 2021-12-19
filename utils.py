from urllib.parse import urlparse
from tqdm.auto import tqdm
import pandas as pd
from spacy.tokens import DocBin


categories = ['technology_science', 'digital_life', 'sports',
              'fashion_beauty_lifestyle', 'money_business', 'politics',
              'people_shows', 'news', 'culture', 'travel', 'music',
              'cars_motors']


def parse_url(url):
    """
    For extracting domains name from full urls
    :param url: str (e.g. 'http://discovermagazine.com/2014/nov/13-y-not')
    :return: sub-domain.second-level-domain.top-level-domain (e.g. 'www.discovermagazine.com')
    """
    return urlparse(url).netloc


def get_input_data(path):
    """
    Read documents and construct documents by concatenating title and body
    :param path: path to input file
    :return: List of (text, label) tuples
    """
    docs = pd.read_csv(path, sep='\t')
    docs['text_label_tuples'] = docs.apply(lambda row: (row['title'] + " " + row["text"], row['category']), axis=1)
    return docs['text_label_tuples'].tolist()


def create_docs(nlp, data):
    """
    Create a list of SpaCy Doc objects from list of raw texts
    :param nlp: SpaCy pipeline to apply on raw texts
    :param data: List of (text, label) tuples
    :return: List of SpaCy Doc objects
    """
    docs = []
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total=len(data)):
        doc.cats = {cat: cat == label for cat in categories}
        docs.append(doc)

    return docs


def save_to_doc_bin(nlp, data, path):
    """
    Process raw texts with a given pipeline, creates a DodBin object and saves it to disk.
    :param nlp: SpaCy pipeline to apply on raw texts
    :param data: List of (text, label) tuples
    :param path: Path to save the DocBin object
    :return: None
    """
    docs = create_docs(nlp, data)
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(path)


def restore_data(path, nlp):
    """
    Reads a collection of docs, saved as binaries on disk, and then recover them using a given Vocab
    :param path: path to DocBin file. Usually with '.spacy' extension
    :param nlp: Spacy pipe
    :return: list of SpaCy Docs
    """
    doc_bin = DocBin().from_disk(path)
    return list(doc_bin.get_docs(nlp.vocab))