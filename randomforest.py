
# Random Forest Algorithm on Blood Spatter Analysis
# Adapted from https://machinelearningmastery.com/implement-random-forest-scratch-python/
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini


# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0]) - 1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del (node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth + 1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root


# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


# Create a random subsample from the dataset with replacement
# stays same
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return (predictions)

if __name__ == '__main__':
	# Test the random forest algorithm
	# random seed
	seed(5)
	# load and prepare data
	# C:\\Users\\annaa\\PycharmProjects\\blood_spatters\\blood_spatters_polygence.csv
	filename = '.\\data\\example.csv'  # .\\data\\sonar.all-data
	dataset = load_csv(filename)
	dataset = dataset[1:]
	for i in range(0, len(dataset)):
		dataset[i] = dataset[i][1:]
	# convert string attributes to float
	for i in range(0, len(dataset[0]) -1):
		str_column_to_float(dataset, i)
	# convert class column to integers
	str_column_to_int(dataset, len(dataset[0]) - 1)  # {'1': 0, '0': 1} switching 0 and 1
	# evaluate algorithm
	# hyperparameters
	n_folds = 5 # number of folds in cross-fold validation
	max_depth = 7 # maximum depth of each decision tree
	min_size = 1 # minimum size of each decision tree
	sample_size = 1.0
	n_features = int(sqrt(len(dataset[0]) - 1))
	list_ntrees = [1, 10, 100]
	list_maxdepth = list(range(1,11))
	results = np.zeros((len(list_ntrees),len(list_maxdepth),n_folds))  # dims = number of trees by number of max depth by number of # folds
	for idx_n_trees, n_trees in enumerate(list_ntrees):  # shows the results for 1, 5, and 30 trees
		for idx_max_depth, max_depth in enumerate(range(1, 11)):
			scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
			print('Trees: %d' % n_trees)
			print('Scores: %s' % scores)
			print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
			results[idx_n_trees, idx_max_depth, :] = np.array(scores)
# plot n_trees
	fig = plt.figure()

	plt.clf()
	plt.plot(list_ntrees, np.mean(results[:, 0, :], axis=1), label = 'mean accuracy')
	ax = fig.axes[0]
	ax.set_xlabel('Number of Trees')
	ax.set_ylabel('Accuracy (%)')
	plt.xscale('log')
	ax.set_xticks(list_ntrees)
	ax.legend()
	fig.savefig('./treesVsAcc.pdf', dpi=300)
	plt.close()
# plot max_depth
	fig = plt.figure()
	plt.clf()
	plt.plot(list_maxdepth, np.mean(results[1, :, :], axis=1), label='mean accuracy')
	ax = fig.axes[0]
	ax.set_xlabel('Max Depth')
	ax.set_ylabel('Accuracy (%)')
	ax.set_xticks(list_maxdepth)
	ax.legend()
	fig.savefig('./maxdepthVsAcc.pdf', dpi=300)
	plt.close()

	# printing results
	plt.scatter(range(0,n_folds), results[0,0,:])


