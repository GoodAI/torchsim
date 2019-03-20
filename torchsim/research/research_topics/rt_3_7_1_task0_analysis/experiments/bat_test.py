import random
import sys

from eval_utils import parse_test_args

if __name__ == '__main__':
    arg = parse_test_args()

    print(f'---------------- arguments are {arg}')

    return_val = random.randint(0, 10)
    print(f'return value {return_val}')

    sys.exit(return_val)

