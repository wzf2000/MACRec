import os
import sys
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks import *

def main():
    init_parser = ArgumentParser()
    init_parser.add_argument('-m', '--main', type=str, required=True, help='The main function to run')
    init_parser.add_argument('--verbose', type=str, default='INFO', choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], help='The log level')
    init_args, init_extras = init_parser.parse_known_args()

    logger.remove()
    logger.add(sys.stderr, level=init_args.verbose)
    os.makedirs('logs', exist_ok=True)
    # log name use the time when the program starts, level is INFO
    logger.add('logs/{time:YYYY-MM-DD:HH:mm:ss}.log', level='DEBUG')

    try:
        task = eval(init_args.main + 'Task')()
    except NameError:
        logger.error('No such task!')
        return
    task.launch()

if __name__ == '__main__':
    main()