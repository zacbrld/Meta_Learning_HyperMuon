import os
import urllib.request
import zipfile

import numpy as np
import tiktoken


def get_wikitext_data(datasets_base_dir):
    """Inspired from https://github.com/tysam-code/hlb-gpt"""
    WIKITEXT_DATA_PATH = os.path.join(datasets_base_dir, "wikitext/")
    train_path = os.path.join(WIKITEXT_DATA_PATH, "train.bin")
    val_path = os.path.join(WIKITEXT_DATA_PATH, "val.bin")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        raw_data_source = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip"
        urllib.request.urlretrieve(
            raw_data_source, os.path.join(WIKITEXT_DATA_PATH, "data.zip")
        )

        with zipfile.ZipFile(
            os.path.join(WIKITEXT_DATA_PATH, "data.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(WIKITEXT_DATA_PATH)

        with open(
            os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.train.raw"), "r"
        ) as data_file:
            raw_train_data = data_file.read()

        with open(
            os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.valid.raw"), "r"
        ) as data_file:
            raw_eval_data = data_file.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
        raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        train_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, "train.bin"))
        eval_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, "val.bin"))
        print("completed the tokenization process!")

    return {
        "train": train_path,
        "val": val_path,
    }
