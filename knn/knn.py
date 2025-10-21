import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score  # ← 增加 PR 指标
plt.switch_backend('Agg')  # 支持无界面环境

# 1. 数据加载与预处理
def load_data(file_path):
    # 直接读取并处理数据（一行完成标签转换）
    data = np.loadtxt(file_path, delimiter='\t',
                      converters={3: lambda x: {'didntLike':0, 'smallDoses':1, 'largeDoses':2}[x.decode()]})
    X, y = data[:, :3], data[:, 3].astype(int)
    
    # 归一化（Min-Max）
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-6)  # 加小值避免除零
    
    # 简单划分训练集（800）和测试集（200）
    return X[200:], X[:200], y[200:], y[:200]


# 2. KNN预测与得分计算
def knn_predict_and_score(X_test, X_train, y_train, k=3):
    y_pred = []
    y_scores = np.zeros((len(X_test), 3))  # 存储3个类别的得分（投票比例）
    
    for i, sample in enumerate(X_test):
        # 计算距离
        distances = np.sqrt(np.sum((X_train - sample)**2, axis=1))
        # 取近邻
        neighbors = y_train[np.argsort(distances)[:k]]
        
        # 预测类别（多数投票）
        y_pred.append(np.bincount(neighbors).argmax())
        
        # 计算每个类别的得分（近邻中该类别的占比，替代概率）
        for c in range(3):
            y_scores[i, c] = np.mean(neighbors == c)
    
    return np.array(y_pred), y_scores


# 3. 绘制ROC曲线（OvR）
def plot_roc(y_test, y_scores, save_path='roc_curve.png'):
    n_classes = 3
    colors = ['red', 'green', 'blue']
    class_names = ['didntLike', 'smallDoses', 'largeDoses']
    
    plt.figure(figsize=(8, 6))
    for i, color, name in zip(range(n_classes), colors, class_names):
        fpr, tpr, _ = roc_curve(y_test == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC of {name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    print(f"ROC曲线已保存至: {save_path}")


# 4. 绘制PR曲线（OvR） 
def plot_pr(y_test, y_scores, save_path='pr_curve.png'):
    n_classes = 3
    colors = ['red', 'green', 'blue']
    class_names = ['didntLike', 'smallDoses', 'largeDoses']
    
    plt.figure(figsize=(8, 6))
    for i, color, name in zip(range(n_classes), colors, class_names):
        precision, recall, _ = precision_recall_curve(y_test == i, y_scores[:, i])
        ap = average_precision_score(y_test == i, y_scores[:, i])  # 面积（AP）
        plt.plot(recall, precision, lw=2, color=color,
                 label=f'PR of {name} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class PR (One-vs-Rest)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    print(f"PR曲线已保存至: {save_path}")


# 主函数
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('datingTestSet.txt')
    print(f"训练集{len(X_train)}，测试集{len(X_test)}")
    
    k = 3
    y_pred, y_scores = knn_predict_and_score(X_test, X_train, y_train, k)
    print(f"准确率: {np.mean(y_pred == y_test):.4f}")
    
    plot_roc(y_test, y_scores)                 # 生成 ROC
    plot_pr(y_test, y_scores)                  # 生成 PR（
