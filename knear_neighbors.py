
import os
from matplotlib import pyplot as plt

import math
import numpy as np

from typing import Tuple, List, Dict

from collections import defaultdict, Counter

# https://scikit-learn.org/stable/datasets.html
from sklearn.datasets import load_iris
import random


def test_train_split(
        data: List,
        prob: float
    ) -> Tuple[List, List]:
    """
    Split data into fractions: [prob, 1-prob]
    """
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


def subtract(
        vector_i: np.array,
        vector_j: np.array
    ) -> np.array:
    """
    Subtracts corresponding elements
    """
    assert len(vector_i) == len(vector_j)
    return [v_i - v_j for v_i, v_j in zip(vector_i, vector_j)]


def dot(
        vector_i: np.array,
        vector_j: np.array
    ) -> float:
    """
    Computes the sum of the product of vectors elements
    """
    assert len(vector_i) == len(vector_j)
    return sum(v_i * v_j for v_i, v_j in zip(vector_i, vector_j))


def sum_of_squares(
        vector: np.array
    ) -> float:
    """
    Sum the squares of each dimension of the vector
    """
    return dot(vector, vector)


def magnitude(
        vector: np.array
    ) -> float:
    """
    Returns the magnitude/length of a vector
    """
    return math.sqrt(sum_of_squares(vector))


def distance(
        vector_i: np.array,
        vector_j: np.array
    ) -> float:
    """
    Computes the distance between vector_i and vector_j
    """
    return magnitude(subtract(vector_i, vector_j))


def major(
        labels: List[str]
    ) -> str:
    """
    Assumes that labels are ordered from nearest to farthest
    """
    counts = Counter(labels)
    major_values, major_count = counts.most_common(1)[0]
    n_majors = len([count for count in counts.values() if count == major_count])
    if n_majors == 1:
        return major_values
    else:
        return major(labels[:-1])


def knn_classify(
        k: int,
        points: List,
        new_point: List
    ) -> str:
    """
    Order the points from nearest to farthest
    Find the labels/class for the k closest
    """
    sorted_points = sorted(points, key=lambda p: distance(p, new_point))
    k_near = [p[-1] for p in sorted_points[:k]]
    return major(k_near)


_this_path = os.path.dirname(__file__)

iris = load_iris()
X = iris.data
y = iris.target
series = iris.target_names
metrics = iris.feature_names

points_by_species = {specie: list() for specie in series}
for i, j in zip(X, y):
    serie = series[j]
    points_by_species[serie].append(i)

pairs = [(i, j) 
         for i in range(len(metrics)) 
         for j in range(len(metrics)) 
         if i < j]  # triangular matrix/system with combinations of metrics

marks = ['+', '.', 'x']

fig, ax = plt.subplots(2, 3)

for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i]} X {metrics[j]}", fontsize=6)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc='lower right', prop={'size': 6})

fig_path = os.path.join(_this_path, 'iris_scatter.png')
plt.savefig(fig_path)
plt.gca().clear()

random.seed(12)

iris_data = [[X[i][k] for k in range(len(metrics))] + [y[i]] for i in range(len(X))]

iris_train, iris_test = test_train_split(iris_data, .7)

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
n_correct = 0
k = 5

for iris in iris_test:
    prediction = knn_classify(k, iris_train, iris)
    iris_class = iris[-1]

    if prediction == iris_class:
        n_correct += 1
    
    confusion_matrix[(prediction, iris_class)] += 1

pct_correct = n_correct / len(iris_test)
print(pct_correct, confusion_matrix)
