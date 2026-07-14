import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, tree
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
def mutil_model(x_train, y_train, x_test, y_test, output_dir,keyname):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scale = scaler.transform(x_train)
    x_test_scale = scaler.transform(x_test)

    # 模型配置
    model_configs = [
        {
            'name': 'SVM',
            'model': SVC(kernel='linear', probability=True),
            'param_grid': {"C": [0.001, 0.01, 0.1, 1, 10],
                           "gamma": [0.0001]},
            'use_scaler': True
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(),
            'param_grid': {'n_estimators': [10, 20, 50, 100],
                           'min_samples_split': range(30, 100, 20),
                           'max_features': range(5, 50, 10),
                           'max_depth': range(6, 21, 3)},
            'use_scaler': False
        },
        {
            'name': 'Gaussian Bayes',
            'model': GaussianNB(),
            'param_grid': None,
            'use_scaler': False
        },
        {
            'name': 'KNN',
            'model': neighbors.KNeighborsClassifier(),
            'param_grid': {'n_neighbors': np.arange(1, 21, 1), 'p': np.arange(1, 11, 1)},
            'use_scaler': True
        },
        {
            'name': 'Decision Tree',
            'model': tree.DecisionTreeClassifier(),
            'param_grid': {'criterion': ['gini', 'entropy'],
                           'max_depth': [10, 20, 30, None],
                           'min_samples_leaf': [1, 2, 3, 5],
                           'min_impurity_decrease': [1e-4, 1e-3, 1e-2, 0.05]},
            'use_scaler': False
        },
        {
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(),
            'param_grid': {'n_estimators': [10, 20, 50, 100],
                           'max_depth': [3, 5, 7],
                           'learning_rate': [0.01, 0.1, 1],
                           'subsample': [0.8, 1.0],
                           'colsample_bytree': [0.8, 1.0]},
            'use_scaler': False
        },
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(max_iter=5000),
            'param_grid': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            'use_scaler': False
        }
    ]

    # 结果存储
    results = []
    all_probabilities = pd.DataFrame()
    all_predictions = pd.DataFrame()
    trained_models = {}

    for config in model_configs:
        name, model, param_grid, use_scaler = config.values()
        print(f"############## {name} modeling ############")
        
        # 选择是否使用标准化
        x_train_to_use = x_train_scale if use_scaler else x_train
        
        # 超参数搜索
        if param_grid:
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=16)
            grid_search.fit(x_train_to_use, y_train)
            best_model = grid_search.best_estimator_
            print(f'Best params: {grid_search.best_params_}, AUC of 10 cv: {grid_search.best_score_}')
        else:
            best_model = model.fit(x_train_to_use, y_train)

        # 保存训练好的模型
        trained_models[name] = best_model

        # 交叉验证预测
        cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_prob = cross_val_predict(best_model, x_train_to_use, y_train, cv=cv_splitter, method='predict_proba')[:, 1]
        #cv_prob = cross_val_predict(best_model, x_train_to_use, y_train, cv=5, method='predict_proba')[:, 1]
        cv_pred = (cv_prob >= 0.5).astype(int)
        
        # 计算指标
        SE, SP, ACC, MCC, AUC = calculate_metrics(y_train, cv_pred, cv_prob)

        # 保存交叉验证结果
        results.append({'Model': name, 'SE': SE, 'SP': SP, 'ACC': ACC, 'MCC': MCC, 'AUC': AUC})
        all_probabilities[name] = cv_prob
        all_predictions[name] = cv_pred

    # 写入交叉验证结果到文件
    pd.DataFrame(results).to_csv(os.path.join(output_dir, f'{keyname}cv_model_results.csv'), index=False)
    all_probabilities.to_csv(os.path.join(output_dir, f'{keyname}cv_predictions.csv'), index=False)
    all_predictions.to_csv(os.path.join(output_dir, f'{keyname}cv_probabilities.csv'), index=False)

    # 测试集预测
    test_results = []
    test_prob = pd.DataFrame()
    test_pred_labels = pd.DataFrame()

    for name, model in trained_models.items():
        use_scaler = next(config['use_scaler'] for config in model_configs if config['name'] == name)
        x_test_to_use = x_test_scale if use_scaler else x_test
        
        pre_label = model.predict(x_test_to_use)
        pre_prob = model.predict_proba(x_test_to_use)[:, 1]

        # 计算测试集的指标
        SE, SP, ACC, MCC, AUC = calculate_metrics(y_test, pre_label, pre_prob)

        # 保存测试集的结果
        test_results.append({'Model': name, 'SE': SE, 'SP': SP, 'ACC': ACC, 'MCC': MCC, 'AUC': AUC})

        # 保存预测概率和标签
        test_prob[name] = pre_prob
        test_pred_labels[name] = pre_label
    
    # 返回模型的预测结果
    model_list = list(zip(trained_models.keys(), trained_models.values()))
    return test_prob, test_pred_labels, model_list, test_results, scaler

def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = accuracy_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_prob)
    MCC = matthews_corrcoef(y_true, y_pred)
    return SE, SP, ACC, MCC, AUC

import os
import pandas as pd
import numpy as np


# =========================================================
# 通用读取
# =========================================================
def load_label(train_file, test_file):
    train = pd.read_csv(train_file, sep="\t")
    test  = pd.read_csv(test_file, sep="\t")

    train = train.rename(columns={'Name': 'ID', 'label': 'Label'})
    test  = test.rename(columns={'Name': 'ID', 'label': 'Label'})

    return train, test


# =========================================================
# 统一训练函数
# =========================================================
def run_model(x_train, x_test, y_train, y_test, outdir, key):

    os.makedirs(outdir, exist_ok=True)

    prob, pred, model_list, results, scaler = mutil_model(
        x_train.values,
        y_train,
        x_test.values,
        y_test,
        outdir,
        key
    )

    prob.to_csv(os.path.join(outdir, f"{key}prob.txt"), sep="\t", index=False)
    pred.to_csv(os.path.join(outdir, f"{key}pred.txt"), sep="\t", index=False)
    pd.DataFrame(results).to_csv(
        os.path.join(outdir, f"{key}metrics.txt"),
        sep="\t",
        index=False
    )

    return results


# =========================================================
# 1. Gene baseline
# =========================================================
def gene_baseline(train_file, test_file, info_file, outdir):

    train_label, test_label = load_label(train_file, test_file)

    info = pd.read_csv(info_file, sep="\t")

    train_df = pd.merge(train_label, info[['ID', 'Gene']], on='ID')
    test_df  = pd.merge(test_label,  info[['ID', 'Gene']], on='ID')

    x_train = pd.get_dummies(train_df['Gene'])
    x_test  = pd.get_dummies(test_df['Gene'])

    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

    y_train = train_df['Label'].values
    y_test  = test_df['Label'].values

    print("\n====== Running Gene baseline ======")
    return run_model(x_train, x_test, y_train, y_test, outdir, "Gene_")


# =========================================================
# 2. Mutation load baseline
# =========================================================
def mutation_baseline(train_file, test_file, pos_file, neg_file, outdir):

    train_label, test_label = load_label(train_file, test_file)

    pos = pd.read_csv(pos_file)
    neg = pd.read_csv(neg_file)

    df = pd.concat([pos, neg])
    mut = df.groupby("protein_id").size().reset_index(name="mutation_load")

    train = pd.merge(train_label, mut, left_on="ID", right_on="protein_id", how="left")
    test  = pd.merge(test_label,  mut, left_on="ID", right_on="protein_id", how="left")

    train["mutation_load"] = train["mutation_load"].fillna(0)
    test["mutation_load"]  = test["mutation_load"].fillna(0)

    x_train = train[["mutation_load"]]
    x_test  = test[["mutation_load"]]

    y_train = train["Label"].values
    y_test  = test["Label"].values

    print("\n====== Running Mutation baseline ======")
    return run_model(x_train, x_test, y_train, y_test, outdir, "Mut_")


# =========================================================
# 3. Sequence length baseline
# =========================================================
def length_baseline(train_file, test_file, info_file, outdir):

    train_label, test_label = load_label(train_file, test_file)

    info = pd.read_csv(info_file, sep="\t")
    info["seq_len"] = info["fasta"].astype(str).apply(len)

    train = pd.merge(train_label, info[['ID', 'seq_len']], on='ID', how='left')
    test  = pd.merge(test_label,  info[['ID', 'seq_len']], on='ID', how='left')

    train["seq_len"] = train["seq_len"].fillna(0)
    test["seq_len"]  = test["seq_len"].fillna(0)

    x_train = train[["seq_len"]]
    x_test  = test[["seq_len"]]

    y_train = train["Label"].values
    y_test  = test["Label"].values

    print("\n====== Running Length baseline ======")
    return run_model(x_train, x_test, y_train, y_test, outdir, "Len_")


# =========================================================
# MAIN：一次性跑完 3个 baseline
# =========================================================
def main():

    train_file = "./2.Training_Data/Lambda/Lambda_trainset-1_label.txt"
    test_file  = "./2.Training_Data/Lambda/Lambda_test_244_label.txt"

    info_file  = "./1.Data/CleanData_info.xls"
    pos_file   = "./1.Data/Pos_mutation_table.csv"
    neg_file   = "./1.Data/Neg_mutation_table.csv"

    outdir = "./2.5_Baseline_model"

    print("\n==============================")
    print("START BASELINE PIPELINE")
    print("==============================\n")

    # 1 Gene
    gene_results = gene_baseline(train_file, test_file, info_file, outdir)

    # 2 Mutation load
    mut_results = mutation_baseline(train_file, test_file, pos_file, neg_file, outdir)

    # 3 Sequence length
    len_results = length_baseline(train_file, test_file, info_file, outdir)

    print("\n==============================")
    print("ALL BASELINE FINISHED")
    print("==============================\n")

    print("Gene:", gene_results)
    print("Mutation:", mut_results)
    print("Length:", len_results)


if __name__ == "__main__":
    main()