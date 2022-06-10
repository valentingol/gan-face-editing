import sys


if __name__ == '__main__':
    arg = sys.argv[1]
    score = float(arg.split('=')[1])
    if score < 2:
        print('rgb(209, 42, 0)')
    elif score < 4:
        print('rgb(209, 129, 0)')
    elif score < 6:
        print('rgb(163, 163, 0)')
    elif score < 8:
        print('rgb(103, 163, 0)')
    else:
        print('rgb(19, 163, 0)')
