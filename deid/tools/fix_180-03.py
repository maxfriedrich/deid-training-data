# Fixes a shift in start/end coordinates that is caused by the special characters in "O’neil’s Court"

import argparse
import os
import re


def fixed_contents(contents):
    result = ''
    edit_here = False
    increment_start = False
    for line in contents:
        if '<LOCATION id="P35"' in line:  # the first broken line still has the correct start but incorrect end
            edit_here = True
        if edit_here:
            if increment_start:
                line = re.sub('start="(\d*)"', lambda x: f'start="{int(x.group(1)) - 4}"', line)
            line = re.sub('end="(\d*)"', lambda x: f'end="{int(x.group(1)) - 4}"', line)
            increment_start = True
        result += line
    return result


def fix_180_03(path):
    with open(path) as f:
        result = fixed_contents(f.read())

    os.rename(path, path + '.bak')
    with open(path, 'w') as f:
        f.write(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='the path to 180-03.xml')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        args.path = os.path.join(args.path, '180-03.xml')
    if not os.path.isfile(args.path):
        raise ValueError(f'Cannot locate 180-03.xml at {os.path.dirname(args.path)}')

    fix_180_03(args.path)


if __name__ == '__main__':
    main()
