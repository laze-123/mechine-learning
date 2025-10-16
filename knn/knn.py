import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  # 导入库函数

plt.switch_backend('Agg')  # 支持无界面环境

# 1. 数据加载与预处理（极简版）
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
    y_scores = np.zeros((len(X_test), 3))  # 存储3个类别的得分
    
    for i, sample in enumerate(X_test):
        # 计算距离
        distances = np.sqrt(np.sum((X_train - sample)**2, axis=1))
        # 取近邻
        neighbors = y_train[np.argsort(distances)[:k]]
        
        # 预测类别
        y_pred.append(np.bincount(neighbors).argmax())
        
        # 计算每个类别的得分（近邻中该类别的占比，替代概率）
        for c in range(3):
            y_scores[i, c] = np.mean(neighbors == c)  # 近邻中属于类别c的比例
    
    return np.array(y_pred), y_scores


# 3. 绘制ROC曲线（用sklearn简化计算）
def plot_roc(y_test, y_scores, save_path='roc_curve.png'):
    n_classes = 3
    colors = ['red', 'green', 'blue']
    class_names = ['didntLike', 'smallDoses', 'largeDoses']
    
    plt.figure(figsize=(8, 6))
    
    # 循环绘制每个类别的ROC曲线（One-vs-Rest）
    for i, color, name in zip(range(n_classes), colors, class_names):
        # 用sklearn计算FPR和TPR（直接替代手动计算）
        fpr, tpr, _ = roc_curve(y_test == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)  # 用sklearn计算AUC
        
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC of {name} (AUC = {roc_auc:.2f})')
    
    # 随机猜测基准线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 图表设置
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC (using sklearn)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    print(f"ROC曲线已保存至: {save_path}")


# 主函数
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('datingTestSet.txt')
    print(f"训练集{len(X_train)}，测试集{len(X_test)}")
    
    k = 3
    y_pred, y_scores = knn_predict_and_score(X_test, X_train, y_train, k)
    print(f"准确率: {np.mean(y_pred == y_test):.4f}")
    
    plot_roc(y_test, y_scores)
    
