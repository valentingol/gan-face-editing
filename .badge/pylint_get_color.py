import sys


if __name__ == '__main__':
    score = float(sys.argv[1])
    if score < 2:
        print('#a30000')
    elif score < 4:
        print('#a33400')
    elif score < 6:
        print('#9ea300')
    elif score < 8:
        print('#6fa300')
    else:
        print('#36a300')
