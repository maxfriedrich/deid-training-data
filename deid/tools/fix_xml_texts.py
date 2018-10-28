import argparse
import os

# noinspection PyProtectedMember
from bs4 import BeautifulSoup, CData


def fix_xml(pred_xml, gold_xml):
    print(pred_xml, gold_xml)
    gold_soup = BeautifulSoup(open(gold_xml, 'r').read(), features='xml')
    gold_text = gold_soup.find('TEXT').string

    print(gold_text.count('\n'))
    
    pred_soup = BeautifulSoup(open(pred_xml, 'r').read(), features='xml')
    pred_soup.find('TEXT').string = CData(gold_text)
    with open(pred_xml, 'w') as f:
        f.write(str(pred_soup))


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

            fix_xml(pred_xml, gold_xml)
    else:
        fix_xml(args.pred, args.gold)


if __name__ == '__main__':
    main()
