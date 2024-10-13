from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import os

def round_two(val):
    return round(val,2)

class ModelApplication:
    def __init__(self, X_train, y_train, X_test, y_test,enable_categorical=True,deployment_mode= False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.enable_categorical = enable_categorical
        if deployment_mode:
            self._cat_disable()

    def _train_xgboost_classifier(self):
        self.xgb_model = xgb.XGBClassifier()
        if self.enable_categorical:
            self.xgb_model.fit(self.X_train, self.y_train, eval_metric= 'aucpr',
                               eval_set=[(self.X_test, self.y_test)], early_stopping_rounds=10)
        else:
            self._cat_disable()
            self.xgb_model.fit(self.X_train, self.y_train, eval_metric= 'aucpr',
                               eval_set=[(self.X_test, self.y_test)], early_stopping_rounds=10)

    def _cat_disable(self):
        cols_cats = ['DOCTOR_SPECIALTY_CODE', 'DOCTOR_CODE', 'DEPARTMENT_TYPE',
                     'PURCHASER_CODE','CONTRACT_NO','TREATMENT_TYPE_INDICATOR','POLICY_CODE']
        for col in cols_cats:
            self.X_train[col] = self.X_train[col].astype(float)
            self.X_test[col]  = self.X_test[col].astype(float)

    def _get_prediction_metrics(self):
        y_pred = self.xgb_model.predict(self.X_test)
        try:
            mod_accuracy = accuracy_score(self.y_test.tolist(), y_pred.tolist(),)
            mod_precision = precision_score(self.y_test.tolist(), y_pred.tolist())
            mod_recall = recall_score(self.y_test.tolist(), y_pred.tolist())
            f1 = (2*mod_precision*mod_recall) / (mod_precision+ mod_recall)
            print('Binary Classifier metrics calculation')
        except:
            mod_accuracy = accuracy_score(self.y_test.tolist(), y_pred.tolist(),)
            mod_precision = precision_score(self.y_test.tolist(), y_pred.tolist(),average='weighted')
            mod_recall = recall_score(self.y_test.tolist(), y_pred.tolist(),average='weighted')
            f1 = (2*mod_precision*mod_recall) / (mod_precision+ mod_recall)
            print('Multi-Class Classifier metrics calculation')

        dict_metrics = {
            "Accuracy": round_two(mod_accuracy),
            "Precision": round_two(mod_precision),
            "Recall": round_two(mod_recall),
            "F1 Score" : round_two(f1)
        }

        return dict_metrics
    def evaluate_model(self):
        xgb_dict = self._get_prediction_metrics()

        return {  "XGBoost": xgb_dict}

    def get_model(self,model_name='outcome',force_retrain=False): ## model_name : [outcome, medical]
        path_model = f'data/utilities/xgboost/model_{model_name}.json'

        if force_retrain:
            self._train_xgboost_classifier()
            self.xgb_model.save_model(path_model)
            print('Training is done')
            return
        if not os.path.exists(path_model) or os.stat(path_model).st_size == 0:
            print('Model not found in path..')
            print('XGBoost Model is training..')
            self._train_xgboost_classifier()
            self.xgb_model.save_model(path_model)
            print('Training is done')

        else:
            print('XGBoost Model is loading from disk..')
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(path_model)

    def model_predict(self,X_batch):
        return self.xgb_model.predict(X_batch)

    def get_feature_importance(self):
        xgb_feats = self.xgb_model.feature_importances_
        return xgb_feats


class ModelApplicationDeployment:
    def __init__(self, X_batch):
        self.X_batch = X_batch
        print('XGBoost Model is loading from disk..')
        self.claims_classifier = xgb.XGBClassifier()
        path_model_class = 'data/utilities/xgboost/model.json'
        self.claims_classifier.load_model(path_model_class)
        print('Model is loaded..')

    def predict(self):
        preds_claims =  self.claims_classifier.predict(self.X_batch)
        return preds_claims

    def interprete(self):
        explainer   = shap.Explainer(self.claims_classifier)
        shap_values = explainer(self.X_batch)
        main_contributing_features = []

        for i in range(self.X_batch.shape[0]):
            shap_row_values = shap_values[i].values
            main_feature_idx = np.argmax(np.abs(shap_row_values))
            main_feature_name = self.X_batch.columns[main_feature_idx]
            main_contributing_features.append(main_feature_name)
        df = pd.DataFrame()
        df['main_contributing_feature'] = main_contributing_features

        return df