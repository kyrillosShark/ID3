import sys
import numpy as np

class Node:    #stores the node class 
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def load_data(filename):
    data = np.loadtxt(filename)
    if data.ndim == 1:  #if the data is 1D (only one example) convert it to 2D
        data = data[np.newaxis, :]  #add a new axis to make the data 2D to store more examples
    return data
def entropy(y):  #calculates entropy
    class_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, feature_index, threshold):
    parent_entropy = entropy(data[:, -1])

    left_split = data[data[:, feature_index] < threshold]
    right_split = data[data[:, feature_index] >= threshold]

    if len(left_split) == 0 or len(right_split) == 0:
        return 0

    n = len(data)
    n_left, n_right = len(left_split), len(right_split)

    ig = parent_entropy - (n_left / n) * entropy(left_split[:, -1]) - (n_right / n) * entropy(right_split[:, -1])
    return ig

def find_best_split(data):  #finds best split using the maximum information gain
    best_gain = 0
    best_split = None
    n_features = data.shape[1] - 1

    for feature_index in range(n_features):
        thresholds = np.unique(data[:, feature_index])
        for threshold in thresholds:
            ig = information_gain(data, feature_index, threshold)
            if ig > best_gain:
                best_gain = ig
                best_split = (feature_index, threshold)

    return best_split

def build_decision_tree(data, depth=0):     #class builds a tree using the data and the the depth of the tree
    X, y = data[:, :-1], data[:, -1]
    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0 or np.unique(y).size == 1:
        leaf_value = np.round(np.mean(y)).astype(int)
        return Node(value=leaf_value)

    feature_index, threshold = find_best_split(data)

    if feature_index is None:
        leaf_value = np.round(np.mean(y)).astype(int)
        return Node(value=leaf_value)

    left_split = data[data[:, feature_index] < threshold]
    right_split = data[data[:, feature_index] >= threshold]

    left_child = build_decision_tree(left_split, depth + 1)
    right_child = build_decision_tree(right_split, depth + 1)

    return Node(feature_index, threshold, left_child, right_child)

def classify(tree, example):          #if the tree is not find then it returns value, otherwise it classifies the left and right side of the recursively 
    if tree.value is not None:
        return tree.value

    if example[tree.feature_index] < tree.threshold:
        return classify(tree.left, example)
    else:
        return classify(tree.right, example)

def main(training_datafile, validation_datafile):  #takes training and validation sets as arguments 
    training_data = load_data(training_datafile)
    validation_data = load_data(validation_datafile)

    decision_tree = build_decision_tree(training_data)

    correct_classifications = 0
    for example in validation_data:
        prediction = classify(decision_tree, example[:-1])
        if prediction == example[-1]:
            correct_classifications += 1

    print(correct_classifications)
main(sys.argv[1], sys.argv[2])

