import time
from random_forests.decision_tree import DecisionTree
from random_forests.utils import Dataset


def main():
    test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
    test_dataset.load_dataset()
    myDecisionTree = DecisionTree()


if __name__ == "__main__":
    main()