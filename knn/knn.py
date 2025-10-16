import numpy as np
import matplotlib.pyplot as plt

# 设置非交互式后端，支持无界面环境
plt.switch_backend('Agg')

# 数据加载与预处理
def load_data(file_path, test_size=0.2, seed=42):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                # 特征转换与标签编码
                features = list(map(float, parts[:3]))
                label_map = {'didntLike':0, 'smallDoses':1, 'largeDoses':2}
                data.append(features + [label_map[parts[3]]])
    
    data = np.array(data)
    X, y = data[:, :3], data[:, 3].astype(int)
    
    # 划分训练集和测试集
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_idx = int(len(X) * test_size)
    X_train, X_test = X[indices[test_idx:]], X[indices[:test_idx]]
    y_train, y_test = y[indices[test_idx:]], y[indices[:test_idx]]
    
    # 特征归一化
    min_vals, max_vals = X_train.min(0), X_train.max(0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    return (X_train - min_vals)/ranges, (X_test - min_vals)/ranges, y_train, y_test


# KNN分类器实现
def knn_predict(X_test, X_train, y_train, k=3):
    y_pred = []
    for sample in X_test:
        # 计算欧氏距离并排序
        distances = np.sqrt(np.sum((X_train - sample)**2, axis=1))
        neighbors = y_train[np.argsort(distances)[:k]]
        # 投票决定类别
        y_pred.append(np.bincount(neighbors).argmax())
    return np.array(y_pred)


# ROC曲线与AUC计算
def roc_auc(X_test, X_train, y_train, y_test, target, k=3):
    # 提取正类样本并计算得分
    pos_train = X_train[y_train == target]
    if len(pos_train) == 0:
        return [], [], 0.0
    
    # 计算每个样本的正类得分
    scores = [1/(np.mean(np.sqrt(np.sum((pos_train - s)**2, axis=1))) + 1e-6) 
             for s in X_test]
    y_true = (y_test == target).astype(int)
    
    # 计算不同阈值下的TPR和FPR
    thresholds = np.unique(scores)
    thresholds = np.concatenate([[max(thresholds)+1], thresholds, [min(thresholds)-1]])
    tpr, fpr = [], []
    
    for thresh in thresholds:
        y_pred = (np.array(scores) >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr.append(tp/(tp+fn) if (tp+fn) > 0 else 0.0)
        fpr.append(fp/(fp+tn) if (fp+tn) > 0 else 0.0)
    
    # 计算AUC
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += 0.5 * (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1])
    
    return np.array(fpr), np.array(tpr), auc


# 绘制ROC曲线（修复了格式字符串错误）
def plot_roc(X_test, X_train, y_train, y_test, k=3, save_path='roc_curve.png'):
    plt.figure(figsize=(8, 6))
    classes = {0: 'didntLike', 1: 'smallDoses', 2: 'largeDoses'}
    colors = ['red', 'green', 'blue']
    
    for target in [0, 1, 2]:
        fpr, tpr, auc = roc_auc(X_test, X_train, y_train, y_test, target, k)
        plt.plot(fpr, tpr, color=colors[target], linewidth=2,
                label=f'{classes[target]} (AUC={auc:.3f})')
    
    # 修复格式字符串：将颜色和线型分开指定
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC=0.5)')
    plt.xlabel('FPR'), plt.ylabel('TPR')
    plt.title(f'KNN ROC Curves (k={k})'), plt.legend()
    plt.grid(alpha=0.3), plt.savefig(save_path, dpi=300)
    print(f"ROC曲线已保存至: {save_path}")


# 主函数
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('datingTestSet.txt')
    print(f"训练集{len(X_train)}个样本，测试集{len(X_test)}个样本")
    
    k = 3
    y_pred = knn_predict(X_test, X_train, y_train, k)
    accuracy = np.mean(y_pred == y_test)
    print(f"KNN准确率 (k={k}): {accuracy:.4f}")
    
    plot_roc(X_test, X_train, y_train, y_test, k)
    
