import argparse

from . import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='the config filename')
    args = parser.parse_args()

    run_experiment(args.config)


if __name__ == '__main__':
    main()
