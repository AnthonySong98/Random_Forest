import time
from decision_tree import DecisionTree
from utils import Dataset


def main():
    test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
    test_dataset.load_dataset()
    myDecisionTree = DecisionTree()
    print("helloworld")


if __name__ == "__main__":
    main()