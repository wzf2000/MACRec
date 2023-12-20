import sys
from loguru import logger
from argparse import ArgumentParser
from reflexion4rec.tasks import *

def main():
    init_parser = ArgumentParser()
    init_parser.add_argument('-m', '--mode', type=str, required=True, help='The main function to run')
    init_parser.add_argument('--verbose', type=str, default='INFO', choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], help='The log level')
    init_args, init_extras = init_parser.parse_known_args()
    logger.remove()
    logger.add(sys.stderr, level=init_args.verbose)
    try:
        task = eval(init_args.mode + 'Task')()
    except NameError:
        logger.error('No such task!')
        return
    task.launch()

if __name__ == '__main__':
    main()