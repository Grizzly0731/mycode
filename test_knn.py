# -*- coding: utf-8 -*-
import random
import numpy as np
from knn import (KNNClassifier, euclidean_distance, manhattan_distance,
                 find_nearest_neighbors)


def test_euclidean_single():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_manhattan_single():
    assert manhattan_distance([0, 0], [3, 4]) == 7.0


def test_euclidean_matrix():
    A = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    B = np.array([[3, 4], [0, 0]], dtype=float)
    expect = np.linalg.norm(A[:, None] - B[None], axis=-1)
    assert np.allclose(euclidean_distance(A, B), expect)


def test_single_vs_matrix():
    d = euclidean_distance([0, 0], [[3, 4], [0, 0], [1, 0]])
    assert d.shape == (3,)
    assert np.allclose(d, [5.0, 0.0, 1.0])


def test_nearest_neighbors():
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.5, 0.0]])
    idx, dist = find_nearest_neighbors(X_train, [0.0, 0.0], k=3)
    assert idx.tolist() == [0, 3, 1]
    assert np.allclose(dist, [0.0, 0.5, 1.0])


def test_vote_majority():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    y_train = np.array(["A", "A", "B", "B", "B"])
    clf = KNNClassifier(k=3).fit(X_train, y_train)
    assert clf.predict([1.2])[0] == "A"


def test_vote_tie():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array(["B", "A", "A", "B"])
    clf = KNNClassifier(k=4).fit(X_train, y_train)
    assert clf.predict([1.5])[0] == "A"


def test_predict_end_to_end():
    X_train = np.array([[0.0], [0.5], [10.0], [10.5]])
    y_train = np.array(["low", "low", "high", "high"])
    clf = KNNClassifier(k=2).fit(X_train, y_train)
    assert clf.predict([[0.2], [10.2]]).tolist() == ["low", "high"]


def test_score():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array(["x", "y"])
    clf = KNNClassifier(k=1).fit(X_train, y_train)
    assert clf.score([[0.1], [0.9]], ["x", "y"]) == 1.0
    assert clf.score([[0.1], [0.9]], ["y", "x"]) == 0.0


def test_reproducible_split():
    a = list(range(20))
    random.seed(42)
    random.shuffle(a)
    b = list(range(20))
    random.seed(42)
    random.shuffle(b)
    assert a == b


TESTS = [
    ("欧氏距离：单样本手算", test_euclidean_single),
    ("曼哈顿距离：单样本手算", test_manhattan_single),
    ("欧氏距离：矩阵与 NumPy 一致", test_euclidean_matrix),
    ("距离函数：单样本 vs 矩阵", test_single_vs_matrix),
    ("最近邻搜索：索引与距离", test_nearest_neighbors),
    ("投票：多数获胜", test_vote_majority),
    ("投票：平票取最小标签", test_vote_tie),
    ("预测：端到端", test_predict_end_to_end),
    ("评分：准确率计算", test_score),
    ("复现：同一个种子结果相同", test_reproducible_split),
]

failed = 0
for name, fn in TESTS:
    try:
        fn()
        print(f"  [通过] {name}")
    except AssertionError:
        failed += 1
        print(f"  [失败] {name}")

print(f"\n{len(TESTS) - failed}/{len(TESTS)} 个测试通过")
if failed:
    raise SystemExit(1)
