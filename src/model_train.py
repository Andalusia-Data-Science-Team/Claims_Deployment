from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier

import xgboost as xgb
import pandas as pd
import numpy as np
import itertools
import logging
import os

logging.getLogger('lightgbm').setLevel(logging.WARNING)

def round_two(val):
    return round(val,2)

def encode_label(labels:list):
    out = []
    for i in range(len(labels)):
        if str(labels[i]).lower().strip() == 'approved':
            out.append(1)
        else:
            out.append(0)
    return out

class ModelsSearchEngine:
    def __init__(self, X_train, y_train, X_test, y_test,enable_categorical=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.enable_categorical = enable_categorical

    def _train_sgd_classifier(self):
        self.sgd_model = SGDClassifier()
        self.sgd_model.fit(self.X_train, self.y_train)
    def _train_decision_tree(self):
        self.dt_model = DecisionTreeClassifier()
        self.dt_model.fit(self.X_train, self.y_train)

    def _train_neural_network(self):
        self.nn_model = MLPClassifier()
        self.nn_model.fit(self.X_train, self.y_train)

    def _train_lightgbm(self):
        self.lgbm_model = LGBMClassifier(force_row_wise=True)
        self.lgbm_model.fit(self.X_train, self.y_train)

    def _train_xgboost_classifier(self):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(self.X_train, self.y_train)

    def _get_prediction_metrics(self,model):
        y_pred = model.predict(self.X_test)
        try:
            mod_accuracy = accuracy_score(self.y_test, y_pred)
            mod_precision = precision_score(self.y_test, y_pred)
            mod_recall = recall_score(self.y_test, y_pred)
            mod_f1 = (2*mod_precision*mod_recall) / (mod_precision+mod_recall)
        except:
            mod_accuracy = accuracy_score(self.y_test, y_pred)
            mod_precision = precision_score(self.y_test, y_pred,average='weighted')
            mod_recall = recall_score(self.y_test, y_pred,average='weighted')
            mod_f1 = (2*mod_precision*mod_recall) / (mod_precision+mod_recall)

        dict_metrics = {
            "Accuracy" : round_two(mod_accuracy),
            "Precision": round_two(mod_precision),
            "Recall"   : round_two(mod_recall),
            "F1 Score" : round_two(mod_f1)}

        return dict_metrics

    def _cat_disable(self):
        cols_cats = ['DOCTOR_SPECIALTY_CODE', 'DOCTOR_CODE', 'DEPARTMENT_TYPE', 'PURCHASER_CODE','CONTRACT_NO','TREATMENT_TYPE_INDICATOR']
        for col in cols_cats:
            self.X_train[col] = self.X_train[col].astype(float)
            self.X_test[col] = self.X_test[col].astype(float)

    def train_models(self):
        if self.enable_categorical == False:
            self._cat_disable()

        self._train_lightgbm()
        self._train_decision_tree()
        self._train_neural_network()
        self._train_sgd_classifier()
        self._train_xgboost_classifier()

        print('\n\nLightGBM, Decision Tree, SGD and Neural Network are trained on dataset.')

    def evaluate_models(self):
        dt_dict  = self._get_prediction_metrics(self.dt_model)
        lgbm_dict= self._get_prediction_metrics(self.lgbm_model)
        nn_dict  = self._get_prediction_metrics(self.nn_model)
        sgd_dict = self._get_prediction_metrics(self.sgd_model)
        xgb_dict = self._get_prediction_metrics(self.xgb_model)

        return {
            "Decision Tree"  : dt_dict,
            "LightGBM"       : lgbm_dict,
            "SGD Classifier" : sgd_dict,
            "XGBoost"        : xgb_dict,
            "Neural Network" : nn_dict
        }

    def get_all_models_names(self):
        evaluation_results = self.evaluate_models()
        all_models = []
        for key in evaluation_results.keys():
            all_models.append(key)
        return all_models

    def get_decision_tree_feature_importance(self):
        return self.dt_model.feature_importances_

    def _get_lightgbm_feature_importance_sub(self):
        return self.lgbm_model.feature_importances_

    def _normalize_feature_importance(self, importance_values):
        total_importance = np.sum(importance_values)
        normalized_importance = importance_values / total_importance
        return normalized_importance

    def get_lightgbm_feature_importance(self):
        lgbm_feature_importance = self._get_lightgbm_feature_importance_sub()
        normalized_importance = self._normalize_feature_importance(lgbm_feature_importance)
        return normalized_importance

    def get_neural_network_feature_importance(self):

        coefficients = self.nn_model.coefs_[0]
        absolute_weights = np.abs(coefficients)

        total_absolute_weight = np.sum(absolute_weights)
        normalized_importance = absolute_weights / total_absolute_weight * 100
        return normalized_importance

    def get_sgd_classifier_feature_importance(self):
        coefficients = self.sgd_model.coef_
        absolute_weights = np.abs(coefficients)

        total_absolute_weight = np.sum(absolute_weights)
        normalized_importance = absolute_weights / total_absolute_weight *100

        return normalized_importance

    def get_xgboost_feature_importance(self):
        return self.xgb_model.feature_importances_

    def get_feature_importance(self):
        dt_feats = self.get_decision_tree_feature_importance()
        lgbm_feats = self.get_lightgbm_feature_importance()
        dnn_feats = self.get_neural_network_feature_importance()
        sgd_feats = self.get_sgd_classifier_feature_importance()
        xgb_feats = self.get_xgboost_feature_importance()

        return dt_feats, lgbm_feats, dnn_feats,sgd_feats, xgb_feats


class XGBoostTuning:
    def __init__(self, X_train, y_train, X_test, y_test,enable_categorical=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = []
        self.enable_categorical = enable_categorical

    def _cat_disable(self):
        cols_cats = ['DOCTOR_SPECIALTY_CODE', 'DOCTOR_CODE', 'DEPARTMENT_TYPE', 'PURCHASER_CODE']
        for col in cols_cats:
            self.X_train[col] = self.X_train[col].astype(float)
            self.X_test[col]  = self.X_test[col].astype(float)

    def train_and_evaluate(self, param_grid):
        if self.enable_categorical == False:
            self._cat_disable()
        param_combinations = list(itertools.product(*param_grid.values()))
        best_score = -1
        best_params = None
        best_model = None

        for combination in param_combinations:
            params = dict(zip(param_grid.keys(), combination))
            try:
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
                model.fit(self.X_train, self.y_train)
            except:
                model = xgb.XGBClassifier(**params, enable_categorical=True,
                                          use_label_encoder=False, eval_metric='logloss')
                model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            print(f"Trying parameters: {params} | F1 Score: {f1:.4f}")
            result_entry = {**params, 'f1_score': f1}
            self.results.append(result_entry)
            if f1 > best_score:
                best_score = f1
                best_params = params
                best_model = model

        self.results_df = pd.DataFrame(self.results)
        self.best_model = best_model
        self.best_params = best_params
        self.best_score = best_score

        return best_model, best_params, best_score

    def save_results(self, file_path='drafts/Reports/2024_09_10/xgboost_parameters.xlsx'):
        if not self.results_df.empty:
            if os.path.exists(file_path):
                df_old = pd.read_excel(file_path)

                df_combined = pd.concat([df_old, self.results_df], ignore_index=True).drop_duplicates()
                df_sorted = df_combined.sort_values(by='f1_score', ascending=False)

                df_sorted.to_excel(file_path, index=False)
                print(f"Results updated in {file_path}.")
            else:
                df_sorted = self.results_df.sort_values(by='f1_score', ascending=False)

                df_sorted.to_excel(file_path, index=False)
                print(f"Results saved to {file_path}.")
        else:
            print("No results to save. Please run train_and_evaluate first.")