import subprocess
from sklearn.model_selection import train_test_split
import spacy
from utils import get_input_data, save_to_doc_bin


def prepare_train_val(raw_data):
    train_val, test = train_test_split(raw_data, test_size=0.15, random_state=1)
    train, val = train_test_split(train_val, test_size=0.1, random_state=1)
    return train, val, test


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_md")
    raw_data = get_input_data('./data/data_redacted.tsv')
    train, val, test = prepare_train_val(raw_data)

    print("preparing data for training...")
    save_to_doc_bin(nlp, train, path="./data/train.spacy")
    save_to_doc_bin(nlp, val, path="./data/valid.spacy")
    save_to_doc_bin(nlp, test, path="./data/test.spacy")

    print("start training a text categorizer...")
    command = "python -m spacy train config.cfg --output ./stam_output"
    p = subprocess.Popen(command.split(" "), cwd="./")
    p.wait()

