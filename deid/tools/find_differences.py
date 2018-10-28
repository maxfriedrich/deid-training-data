import argparse
import os

from bs4 import BeautifulSoup


def print_differences(pred_xml, gold_xml):
    print(os.path.basename(pred_xml))

    sets = []
    for thing in [pred_xml, gold_xml]:
        soup = BeautifulSoup(open(thing, 'r').read(), 'xml')

        items = set()
        for tag in soup.find('TAGS').findChildren():
            items.add((tag.get('TYPE'), tag.get('start'), tag.get('end'), tag.get('text')))
        sets.append(items)

    false_positives = sets[0] - sets[1]
    if len(false_positives) > 0:
        print('  false positives:')
        for fp in false_positives:
            print('  -', fp)

    false_negatives = sets[1] - sets[0]
    if len(false_negatives) > 0:
        print('  false negatives:')
        for fn in false_negatives:
            print('  -', fn)

    print('-' * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred',
                        help='the predictions file or directory')
    parser.add_argument('gold', help='the gold file or directory')

    args = parser.parse_args()

    if os.path.isdir(args.pred):
        for xml in [f for f in os.listdir(args.pred) if f.endswith('.xml')]:
            pred_xml = os.path.join(args.pred, xml)
            gold_xml = os.path.join(args.gold, xml)

            print_differences(pred_xml, gold_xml)
    else:
        print_differences(args.pred, args.gold)


if __name__ == '__main__':
    main()
