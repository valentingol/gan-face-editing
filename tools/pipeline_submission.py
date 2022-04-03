import os

if __name__ == '__main__':
    os.system('FORCE_NATIVE=1 python3 tools/translate.py')
    os.system('python3 tools/preprocess/mixup.py')
    os.system('python3 tools/preprocess/segment.py')
