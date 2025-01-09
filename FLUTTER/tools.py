import sys


param = dict(
GREY = "\033[1;30m",
RED = "\033[1;31m",
YELLOW = "\033[1;33m",
BLUE = "\033[1;34m",
CYAN = "\033[1;36m",
GREEN = "\033[0;32m",
RESET = "\033[0;0m",
BOLD = "\033[;1m",
REVERSE = "\033[;7m")


def set_section(message, color='GREEN'):
    sys.stdout.write(param[color])
    line()
    print(message)
    line()
    sys.stdout.write(param['RESET'])


def line():
    print('=' * 50)
