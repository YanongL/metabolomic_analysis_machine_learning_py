import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import levene, ttest_ind
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


# ======================= 特征选择函数 =======================
def ttest_feature_selection(x_train, y_train, x_test, x_external=None, feature_names=None, alpha=0.05):
    """T检验特征选择（增加外部测试集处理）"""
    zero_idx = np.where(y_train == 0)[0]
    one_idx = np.where(y_train == 1)[0]
    selected_features = []
    p_values = []
    t_stats = []
    selected_indices = []  # 保存所选特征的索引

    for i in range(x_train.shape[1]):
        # 方差齐性检验
        _, p_levene = levene(x_train[zero_idx, i], x_train[one_idx, i])
        # T检验
        t_stat, p_val = ttest_ind(x_train[zero_idx, i], x_train[one_idx, i],
                                  equal_var=(p_levene > 0.05))
        if p_val < alpha:
            selected_indices.append(i)  # 保存特征索引
            selected_features.append(i)
            p_values.append(p_val)
            t_stats.append(abs(t_stat))

    x_train_selected = x_train[:, selected_indices]
    x_test_selected = x_test[:, selected_indices]

    if x_external is not None:
        x_external_selected = x_external[:, selected_indices]
    else:
        x_external_selected = None

    selected_feature_names = None
    if feature_names is not None:
        selected_feature_names = [str(feature_names[i]) for i in selected_indices]  # Convert to string here

    return x_train_selected, x_test_selected, x_external_selected, selected_feature_names, p_values, t_stats, selected_indices


def has_feature_importance(estimator):
    """检查估计器是否支持特征重要性"""
    return hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'coef_')


def lasso_feature_selection_with_cv(X_train, y_train, X_test, selected_feature_names, cv):
    """使用网格搜索优化的LASSO进行特征选择"""
    print("正在使用LASSO进行特征选择，并优化超参数...")

    # LASSO的超参数网格
    lasso_param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    }

    # 创建LASSO回归模型
    lasso = Lasso(max_iter=2000, random_state=42)

    # 使用网格搜索找到最佳的alpha值
    lasso_grid_search = GridSearchCV(
        estimator=lasso,
        param_grid=lasso_param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',  # 对于回归问题使用MSE
        n_jobs=-1,
        verbose=1
    )

    # 拟合网格搜索
    lasso_grid_search.fit(X_train, y_train)

    # 获取最佳的LASSO模型
    best_lasso = lasso_grid_search.best_estimator_
    best_alpha = lasso_grid_search.best_params_['alpha']

    print(f"LASSO最佳alpha参数: {best_alpha}")

    # 使用SelectFromModel进行特征选择
    # 设置阈值为"mean"，选择系数绝对值大于平均值的特征
    lasso_selector = SelectFromModel(
        best_lasso,
        threshold='mean',
        prefit=True
    )

    # 应用特征选择
    X_train_lasso = lasso_selector.transform(X_train)
    X_test_lasso = lasso_selector.transform(X_test)

    # 获取选择的特征名
    feature_mask = lasso_selector.get_support()
    lasso_selected_features = [selected_feature_names[i] for i in range(len(selected_feature_names)) if feature_mask[i]]

    # 如果选择的特征太少，降低阈值
    if len(lasso_selected_features) < 5:
        print(f"LASSO选择的特征太少({len(lasso_selected_features)})，降低阈值重新选择...")
        lasso_selector_lower = SelectFromModel(
            best_lasso,
            threshold='median',  # 使用中位数作为阈值
            prefit=True
        )
        X_train_lasso = lasso_selector_lower.transform(X_train)
        X_test_lasso = lasso_selector_lower.transform(X_test)
        feature_mask = lasso_selector_lower.get_support()
        lasso_selected_features = [selected_feature_names[i] for i in range(len(selected_feature_names)) if
                                   feature_mask[i]]

        # 如果还是太少，至少保留前10个最重要的特征
        if len(lasso_selected_features) < 5:
            print("使用前10个最重要的特征...")
            # 获取LASSO系数的绝对值
            lasso_coefs = np.abs(best_lasso.coef_)
            # 获取前10个最大系数的索引
            top_indices = np.argsort(lasso_coefs)[-10:]
            X_train_lasso = X_train[:, top_indices]
            X_test_lasso = X_test[:, top_indices]
            lasso_selected_features = [selected_feature_names[i] for i in top_indices]

    print(f"LASSO特征选择完成，保留特征数量: {len(lasso_selected_features)}")

    return X_train_lasso, X_test_lasso, lasso_selected_features, best_alpha


def apply_feature_selection(model_name, best_model, X_train, y_train, X_test, selected_feature_names, cv):
    """根据模型类型应用不同的特征选择方法"""

    if model_name == 'SVM':
        # 对SVM使用LASSO特征选择
        print(f"对 {model_name} 使用 LASSO 进行特征选择...")
        X_train_final, X_test_final, final_feature_names, best_alpha = lasso_feature_selection_with_cv(
            X_train, y_train, X_test, selected_feature_names, cv
        )
        print(f"LASSO最佳alpha: {best_alpha}")

    elif has_feature_importance(best_model) or model_name == 'Logistic Regression':
        # 对于支持特征重要性的模型，使用RFECV
        print(f"对 {model_name} 使用 RFECV 进行特征选择...")

        # 对于逻辑回归，确保使用支持coef_的参数
        if model_name == 'Logistic Regression':
            # 创建一个支持coef_的逻辑回归模型用于RFECV
            lr_for_rfecv = LogisticRegression(
                C=best_model.get_params().get('C', 1.0),
                penalty='l2',  # 确保使用l2正则化，支持coef_
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            estimator_for_rfecv = lr_for_rfecv
        else:
            estimator_for_rfecv = best_model

        try:
            rfecv = RFECV(
                estimator=estimator_for_rfecv,
                step=1,
                cv=cv,
                scoring='roc_auc',
                min_features_to_select=max(1, len(selected_feature_names) // 10),  # 至少保留10%的特征
                n_jobs=-1,
                verbose=1
            )

            rfecv.fit(X_train, y_train)

            # 获取选择的特征
            feature_mask = rfecv.support_
            X_train_final = X_train[:, feature_mask]
            X_test_final = X_test[:, feature_mask]
            final_feature_names = [selected_feature_names[i] for i in range(len(selected_feature_names)) if
                                   feature_mask[i]]

            print(f"RFECV后保留的特征数量: {len(final_feature_names)}")

        except Exception as e:
            print(f"RFECV失败: {e}")
            print(f"对 {model_name} 使用 SelectKBest 作为备选方案...")
            # 使用SelectKBest作为备选
            k = min(50, X_train.shape[1])  # 选择最多50个特征
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_final = selector.fit_transform(X_train, y_train)
            X_test_final = selector.transform(X_test)
            feature_mask = selector.get_support()
            final_feature_names = [selected_feature_names[i] for i in range(len(selected_feature_names)) if
                                   feature_mask[i]]

    else:
        # 对于其他不支持特征重要性的模型，使用SelectKBest
        print(f"对 {model_name} 使用 SelectKBest 进行特征选择...")

        # 根据数据集大小选择合适的特征数量
        if X_train.shape[1] > 100:
            k = min(50, X_train.shape[1] // 2)
        else:
            k = min(20, X_train.shape[1] // 2)

        k = max(k, 5)  # 至少保留5个特征

        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_final = selector.fit_transform(X_train, y_train)
        X_test_final = selector.transform(X_test)

        # 获取选择的特征名
        feature_mask = selector.get_support()
        final_feature_names = [selected_feature_names[i] for i in range(len(selected_feature_names)) if feature_mask[i]]

        print(f"SelectKBest后保留的特征数量: {len(final_feature_names)}")

    return X_train_final, X_test_final, final_feature_names


# 主函数
def main():
    # 1. 读取数据
    file_path = r"D:\Yanong_0620\new_0620.xlsx"
    data = pd.read_excel(file_path)

    # 2. 分离特征和标签
    X = data.drop(['ID', 'Label'], axis=1)
    y = data['Label']

    # 确保DataFrame的列名全部为字符串类型
    X.columns = X.columns.astype(str)
    feature_names = X.columns.tolist()

    # 3. 使用StratifiedShuffleSplit进行分层抽样 (80% 训练, 20% 测试)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 4. 特征标准化 (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. T检验特征选择
    X_train_selected, X_test_selected, _, selected_feature_names, p_values, t_stats, selected_indices = ttest_feature_selection(
        X_train_scaled, y_train.values, X_test_scaled, None, feature_names
    )

    print(f"T检验后保留的特征数量: {len(selected_feature_names)}")
    print(f"T检验选择的特征:")
    for i, feature in enumerate(selected_feature_names):
        print(f"{i + 1}. {feature} (p-value: {p_values[i]:.6f}, t-statistic: {t_stats[i]:.4f})")

    # 6. 为每个分类模型定义参数网格
    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        },
        'XGBoost': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    }

    # 基础模型
    base_models = {
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    # 使用StratifiedKFold进行分层5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储结果的字典
    results = {}
    best_params = {}
    final_selected_features = {}  # 存储每个模型最终选择的特征
    feature_selection_info = {}  # 存储特征选择的详细信息

    # 7. 训练并评估每个模型
    for name, model in base_models.items():
        print(f"\n{'=' * 50}")
        print(f"正在优化 {name} 模型...")
        print(f"{'=' * 50}")

        # 使用GridSearchCV找到最佳参数
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # 拟合网格搜索模型
        grid_search.fit(X_train_selected, y_train)

        # 获取最佳参数并存储
        best_params[name] = grid_search.best_params_
        print(f"{name} 最佳参数: {best_params[name]}")

        # 获取最佳模型
        best_model = grid_search.best_estimator_

        # 应用特征选择
        X_train_final, X_test_final, final_feature_names = apply_feature_selection(
            name, best_model, X_train_selected, y_train, X_test_selected, selected_feature_names, cv
        )

        # 添加前缀 "m/z_"
        final_feature_names_prefixed = ["m/z_" + str(feature) for feature in final_feature_names]
        final_selected_features[name] = final_feature_names_prefixed

        # 存储特征选择信息
        feature_selection_info[name] = {
            'method': 'LASSO with GridSearch' if name == 'SVM' else (
                'RFECV' if has_feature_importance(best_model) or name == 'Logistic Regression' else 'SelectKBest'),
            'features_count': len(final_feature_names),
            'original_count': len(selected_feature_names)
        }

        # 重新训练最佳模型（使用最终选择的特征）
        best_model.fit(X_train_final, y_train)

        # 对测试集进行预测
        y_test_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
        y_test_pred = best_model.predict(X_test_final)

        # 对训练集进行预测
        y_train_pred_proba = best_model.predict_proba(X_train_final)[:, 1]
        y_train_pred = best_model.predict(X_train_final)

        # 计算测试集的AUC和ACC
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_acc = accuracy_score(y_test, y_test_pred)

        # 计算训练集的AUC和ACC
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_acc = accuracy_score(y_train, y_train_pred)

        # 存储结果
        results[name] = {
            'Test AUC': test_auc,
            'Test ACC': test_acc,
            'Train AUC': train_auc,
            'Train ACC': train_acc,
            'Features Count': len(final_feature_names),
            'Feature Selection Method': feature_selection_info[name]['method']
        }

        # 打印结果
        print(f"\n{name} 最终结果:")
        print(f"  特征选择方法: {feature_selection_info[name]['method']}")
        print(f"  特征数量: {len(final_feature_names)} (原始: {len(selected_feature_names)})")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test ACC: {test_acc:.4f}")
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Train ACC: {train_acc:.4f}")

    # 8. 打印总结结果
    print(f"\n{'=' * 80}")
    print("模型性能汇总")
    print(f"{'=' * 80}")
    print(f"{'模型':<20} {'特征选择方法':<20} {'Test AUC':<10} {'Test ACC':<10} {'特征数':<8}")
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f"{name:<20} {metrics['Feature Selection Method']:<20} {metrics['Test AUC']:<10.4f} {metrics['Test ACC']:<10.4f} {metrics['Features Count']:<8}")

    # 9. 保存每个模型选择的特征到Excel文件
    try:
        with pd.ExcelWriter(r'D:\Yanong_0620\selected_features.xlsx') as writer:
            for model_name, features in final_selected_features.items():
                # 创建DataFrame，包含特征选择方法信息
                df = pd.DataFrame({
                    'Feature': features,
                    'Selection_Method': [feature_selection_info[model_name]['method']] * len(features)
                })
                # 保存到Excel的指定sheet
                df.to_excel(writer, sheet_name=model_name, index=False)

        print(f"\n已将各模型选择的特征保存到 'D:\\Yanong_0620\\selected_features.xlsx'")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

    return results, best_params, final_selected_features, selected_feature_names, feature_selection_info


if __name__ == "__main__":
    results, best_params, final_features, t_test_features, fs_info = main()