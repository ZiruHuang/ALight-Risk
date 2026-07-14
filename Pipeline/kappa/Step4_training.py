import shutil
import os
import pandas as pd
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve, roc_auc_score
import pandas as pd
import math
import time
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def prof_evalue(true,pred):
    N_pos = 0
    N_neg = 0
    TP = 0
    TN = 0
    for i in range(len(true)):
        if true[i]==1:
            N_pos += 1
            if pred[i]==1:
                TP += 1
        else:
            if true[i]==0:
                N_neg += 1
                if pred[i]==0:
                    TN += 1
    FP = N_pos-TP
    FN = N_neg-TN
    SE = TP/N_pos
    SP = TN/N_neg
    ACC = (TP + TN)/(N_pos + N_neg)
    if (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) == 0:
        MCC = 0
    else:
        MCC = (TN * TP - FP * FN)/math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return SE,SP,ACC,MCC

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
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = accuracy_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_prob)
    MCC = matthews_corrcoef(y_true, y_pred)
    return SE, SP, ACC, MCC, AUC

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

def pro_test_dimention(test,reduce_file):
    df1 = pd.read_csv(test, sep = ',')
    df2 = pd.read_csv(reduce_file, sep = ',')
    df3 = df1[df2.keys()]
    return df3

### 训练Kappa
for i in range(1, 2):  # 遍历 subModel_1 到 subModel_7
    trainpath = f'5.MRMD_Result/Kappa/trainset_{i}/'
    outpath = f'6.Model/Kappa/trainset_{i}/'
    for file in os.listdir(trainpath):
        if file.endswith('_reduce.csv'):
            feature_name = file.split('_')[2]
            print(f'============================ use {feature_name} from trainset_{i} =============================')

            # 读取训练集
            df_train = pd.read_csv(os.path.join(trainpath, file), index_col=None)
            x_train = df_train.iloc[:, 1:].values
            y_train = df_train.iloc[:, 0].values

            # 读取测试集并统一维度
            test_file = f'3.iFeature/Kappa/test/Kappa_test_88_{feature_name}.csv'
            df_test = pro_test_dimention(test_file, os.path.join(trainpath, file))
            x_test = df_test.iloc[:, 1:].values
            y_test = df_test.iloc[:, 0].values

            # 模型输出目录
            modelpath = os.path.join(outpath, f'trainset_{i}{feature_name}_models')
            os.makedirs(modelpath, exist_ok=True)

            # 模型训练与评估
            prob, pred_labels, model_list, mutil_model_test, scaler = mutil_model(
                x_train, y_train, x_test, y_test, modelpath,f'trainset_{i}{feature_name}_'
            )

            # 保存预测结果
            prob.to_csv(os.path.join(modelpath, f'trainset_{i}{feature_name}_prob.txt'), sep='\t', index=None)
            pred_labels.to_csv(os.path.join(modelpath, f'trainset_{i}{feature_name}_pred_label.txt'), sep='\t', index=None)
            pd.DataFrame(mutil_model_test).to_csv(os.path.join(modelpath, f'trainset_{i}{feature_name}_mutil_model_test.txt'), sep='\t', index=None)

            # 保存模型与标准化器
            for model_name, model_obj in model_list:
                model_file = os.path.join(modelpath, f'trainset_{i}{feature_name}_{model_name}.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(model_obj, f)
                    
            joblib.dump(scaler, os.path.join(modelpath, f'trainset_{i}{feature_name}_scaler.pkl'))




#### 训练lambda



import os
import pickle
import pandas as pd

def model_predict(x_test, y_test, model_dir, scaler):
    # 标准化测试数据
    x_test_scale = scaler.transform(x_test)

    # 测试集预测
    test_results = []
    test_prob = pd.DataFrame()
    test_pred_labels = pd.DataFrame()

    # 模型文件列表（忽略scaler和非pkl）
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'scaler' not in f]

    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        model_path = os.path.join(model_dir, model_file)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 判断是否使用标准化
        use_scaler = True if model_name in ['SVM', 'KNN'] else False
        x_test_to_use = x_test_scale if use_scaler else x_test

        try:
            prob = model.predict_proba(x_test_to_use)[:, 1]
            pred = model.predict(x_test_to_use)

            SE, SP, ACC, MCC, AUC = calculate_metrics(y_test, pred, prob)
            test_results.append({
                'Model': model_name,
                'SE': SE,
                'SP': SP,
                'ACC': ACC,
                'MCC': MCC,
                'AUC': AUC
            })

            test_prob[model_name] = prob
            test_pred_labels[model_name] = pred

        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            continue

    return test_prob, test_pred_labels, test_results

## 预测




## 预测
for i in range(1, 2):  # 遍历 subModel_1 到 subModel_7
    trainpath = f'5.MRMD_Result/Kappa/trainset_{i}/'
    modelpath = f'6.Model/Kappa/trainset_{i}/'
    outpath = f'7.TrainingScore/Kappa/trainset_{i}/'
    for file in os.listdir(trainpath):
        os.makedirs(outpath, exist_ok=True)
        if file.endswith('_reduce.csv'):
            feature_name = file.split('_')[2]
            print(f'============================ use {feature_name} from subModel_{i} =============================')

            # 读取训练集
            df_train = pd.read_csv(os.path.join(trainpath, file), index_col=None)
            x_train = df_train.iloc[:, 1:].values
            y_train = df_train.iloc[:, 0].values

            # 读取测试集并统一维度
            test_file = f'3.iFeature/Kappa/All_traning_data/Kappa_trainset-1_{feature_name}.csv'
            df_test = pro_test_dimention(test_file, os.path.join(trainpath, file))
            x_test = df_test.iloc[:, 1:].values
            y_test = df_test.iloc[:, 0].values

            # 评估
            scaler_path = os.path.join(modelpath, f'trainset_{i}{feature_name}_models',f'trainset_{i}{feature_name}_scaler.pkl')   
            scaler = joblib.load(scaler_path)
            model_dir = os.path.join(modelpath,f'trainset_{i}{feature_name}_models')
            prob, pred, mutil_model_test = model_predict(
              x_test, y_test, model_dir,scaler
            )

            # 保存预测结果
            prob.to_csv(os.path.join(outpath, f'trainset_{i}{feature_name}_prob.txt'), sep='\t', index=None)
            pred.to_csv(os.path.join(outpath, f'trainset_{i}{feature_name}_pred_label.txt'), sep='\t', index=None)
            pd.DataFrame(mutil_model_test).to_csv(os.path.join(outpath, f'trainset_{i}{feature_name}_mutil_model_test.txt'), sep='\t', index=None)
