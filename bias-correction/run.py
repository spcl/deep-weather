from cross_validate import cross_validate
from config import getFlag, print_flag


def run(modelname, of):
    import sys
    sys.stdout = open(of, 'w')
    FLAGS = getFlag(modelname)
    print_flag(FLAGS)
    return cross_validate(modelname, FLAGS)

if __name__ == "__main__":
    modelname = "LCN"
    run(modelname, "result.txt")

