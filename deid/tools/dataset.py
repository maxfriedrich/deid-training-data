import argparse
import os
import random
import shutil
from datetime import datetime

from typing import NamedTuple

from ..env import env

NUM_TRAIN_VALID_DOCS = 790  # the total number of train + validation docs in the i2b2 dataset


class Document(NamedTuple):
    csv: str
    txt: str
    xml: str


def make_dataset(train_split: float, valid_split: float):
    data_dir = os.path.join(env.data_dir, 'generated')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    config = f'{train_split}-{valid_split}-{date_str}'

    dataset_train_dir = os.path.join(data_dir, f'train-{config}')
    os.mkdir(dataset_train_dir)

    dataset_train_xml_dir = os.path.join(data_dir, f'train-{config}_xml')
    os.mkdir(dataset_train_xml_dir)

    dataset_valid_dir = os.path.join(data_dir, f'validation-{config}')
    os.mkdir(dataset_valid_dir)

    dataset_valid_xml_dir = os.path.join(data_dir, f'validation-{config}_xml')
    os.mkdir(dataset_valid_xml_dir)

    all_documents = []
    for dataset in ['train', 'validation']:
        dataset_dir = os.path.join(env.data_dir, dataset)
        dataset_xml_dir = os.path.join(env.data_dir, dataset + '_xml')
        for filename in [filename for filename in os.listdir(dataset_dir) if filename.endswith('csv')]:
            csv_filename = os.path.join(dataset_dir, filename)
            txt_filename = os.path.join(dataset_dir, filename[:-3] + 'txt')
            xml_filename = os.path.join(dataset_xml_dir, filename[:-3] + 'xml')
            all_documents.append(Document(csv=csv_filename, xml=xml_filename, txt=txt_filename))

    size = min(max(int(train_split * NUM_TRAIN_VALID_DOCS), 2), NUM_TRAIN_VALID_DOCS)
    train_documents = random.sample(all_documents, size)
    valid_size = max(int(valid_split * len(train_documents)), 1)
    valid_documents = random.sample(train_documents, valid_size)
    print(f'Using {size-valid_size} train documents and {valid_size} validation documents.')

    for document in train_documents:
        target = dataset_valid_dir if document in valid_documents else dataset_train_dir
        shutil.copy2(document.csv, target)
        shutil.copy2(document.txt, target)
        shutil.copy2(document.xml, target + '_xml')

    print(f'Made dataset at {dataset_train_dir}, {dataset_valid_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.description = 'Make train and validation sets of a specified size'
    parser.add_argument('train_split', type=float)
    parser.add_argument('--valid_split', type=float, default=0.2)
    args = parser.parse_args()

    make_dataset(args.train_split, args.valid_split)


if __name__ == '__main__':
    main()
