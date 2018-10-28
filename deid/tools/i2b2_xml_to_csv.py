# Call this using `python -m deid.tools.i2b2_xml_to_csv [params]

import argparse
import csv
import html
import os
from typing import Tuple, Sequence

from bs4 import BeautifulSoup
from tqdm import tqdm

from ..data import Token, tokenize


def xml_to_csv(filename: str, output_dir: str, check_alignment) -> None:
    tokens, text = xml_to_annotated_tokens_and_text(filename, check_alignment)

    path_without_ext = os.path.join(output_dir, os.path.basename(filename)[:-4])

    with open(path_without_ext + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'type', 'start', 'end'])
        writer.writerows(tokens)

    with open(path_without_ext + '.txt', 'w') as f:
        f.write(text)


def xml_to_annotated_tokens_and_text(filename, check_alignment) -> Tuple[Sequence[Token], str]:
    soup = BeautifulSoup(open(filename).read(), features='xml')

    text = str(soup.find('TEXT').contents[0])
    tags = soup.find('TAGS').findChildren()

    if check_alignment:
        # Sanity check: compare the <tag text='...'> with the above text.
        # Ignoring differences where only a '\n' is missing from the tag text because this occurs often in the data
        # and does not seem to matter for us.
        for tag in tags:
            tag_text, original_text = tag.get('text'), text[int(tag.get('start')):int(tag.get('end'))]
            if tag_text != original_text and tag_text != original_text.replace('\n', ' '):
                location = f"{os.path.basename(filename)}[{tag.get('start')}:{tag.get('end')}]"
                tqdm.write(f"{location} (tag) {tag_text.__repr__()} ≠ (original) {original_text.__repr__()}")

        # TODO check here if the start and end tags actually fall on tokens

    doc = tokenize(text)
    return annotate_with_tags(doc, tags), text


def annotate_with_tags(doc, tags) -> Sequence[Token]:
    def tag_start(i):
        return int(tags[i].get('start'))

    def tag_end(i):
        return int(tags[i].get('end'))

    def is_current_tag(token):
        if token.idx == tag_start(current_tag):
            return True
        if token.idx >= tag_start(current_tag) and token.idx + len(token.text) <= tag_end(current_tag):
            return True
        if token.idx < tag_start(current_tag) < token.idx + len(token.text):
            return True
        return False

    current_tag = 0

    result = []
    for sentence in doc.sents:
        continue_tag_type = None  # set to the tag type string if the tag is not yet processed fully
        result.append(Token('<s>', 'O', sentence[0].idx, sentence[0].idx))

        for token in sentence:
            if continue_tag_type and token.idx < tag_end(current_tag):
                tag = f'I-{continue_tag_type}'
            else:
                if token.idx >= tag_end(current_tag) and current_tag < len(tags) - 1:
                    current_tag += 1

                # make sure we did not skip an entire tag
                while token.idx >= tag_end(current_tag) and current_tag < len(tags) - 1:
                    print('Skipping a tag:', tags[current_tag].get('TYPE'), tags[current_tag].get('text'))
                    current_tag += 1

                if is_current_tag(token):
                    continue_tag_type = tags[current_tag].get('TYPE')
                    tag = f'B-{continue_tag_type}'
                else:
                    tag = 'O'
                    continue_tag_type = None

            token_text = token._.unescaped_html if token._.unescaped_html is not None else token.text
            token_text = token_text.strip()
            if len(token_text) == 0 and tag.startswith('B'):
                continue_tag_type = None

            if len(token_text) != 0:
                result.append(Token(token_text, tag, token.idx, token.idx + len(token)))

        result.append(Token('</s>', 'O', sentence[-1].idx, sentence[-1].idx))

    return result


def main() -> None:
    def ensure_file_or_dir(arg) -> str:
        if type(arg) == str and (os.path.isfile(arg) or os.path.isdir(arg)):
            return arg
        raise argparse.ArgumentTypeError(f'{arg} is not a file or directory')

    def ensure_dir(arg) -> str:
        if type(arg) == str and os.path.isdir(arg):
            return arg
        raise argparse.ArgumentTypeError(f'{arg} is not a directory')

    parser = argparse.ArgumentParser()
    parser.add_argument('file_or_dir', help='the input file(s)', type=ensure_file_or_dir)
    parser.add_argument('output_dir', help='save the csv file(s) here', type=ensure_dir)
    parser.add_argument('--check', help='check the tag/text alignment', action='store_true')
    args = parser.parse_args()

    if os.path.isdir(args.file_or_dir):
        filenames = sorted([file for file in os.listdir(args.file_or_dir) if file.endswith('.xml')])
        if len(filenames) == 0:
            print('No XML files found.')

        pbar = tqdm(filenames)
        for filename in pbar:
            pbar.set_description(filename)
            path = os.path.join(args.file_or_dir, filename)
            xml_to_csv(path, output_dir=args.output_dir, check_alignment=args.check)
    else:
        xml_to_csv(args.file_or_dir, output_dir=args.output_dir, check_alignment=args.check)


if __name__ == '__main__':
    main()
