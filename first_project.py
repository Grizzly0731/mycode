import matplotlib
matplotlib.use("Agg")          # 只存图，不弹窗

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 解决中文显示成方块的问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 拿数据：鸢尾花，机器学习界的"Hello World"
iris = load_iris()
X, y = iris.data, iris.target
print("数据量:", X.shape[0], "朵花，每朵", X.shape[1], "个特征")
print("类别:", list(iris.target_names))

# 2. 分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("训练用:", len(X_train), "朵   测试用:", len(X_test), "朵")

# 3. 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 考它
acc = accuracy_score(y_test, model.predict(X_test))
print(f"准确率: {acc:.1%}")

# 5. 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

colors = ["#e74c3c", "#2ecc71", "#3498db"]
for i, name in enumerate(iris.target_names):
    ax1.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=name, alpha=0.8)
ax1.set_xlabel("花萼长度 (cm)")
ax1.set_ylabel("花萼宽度 (cm)")
ax1.set_title("三种鸢尾花的分布")
ax1.legend()

names = ["花萼长", "花萼宽", "花瓣长", "花瓣宽"]
ax2.barh(names, model.feature_importances_, color="#9b59b6")
ax2.set_xlabel("重要性")
ax2.set_title("哪个特征最能区分品种")

plt.tight_layout()
plt.savefig("iris_result.png", dpi=110)
print("图已保存: iris_result.png")
