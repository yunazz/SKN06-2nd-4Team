
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib

ohe_encoder_loaded = joblib.load('models/ohe_encoder.pkl')

# ProportionalImputer - 사용자 정의 imputer
class ProportionalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.fill_values = {}

    def fit(self, X, y=None):
        for column in self.columns:
            value_counts = X[column].value_counts(normalize=True)
            self.fill_values[column] = (value_counts.index, value_counts.values)
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:

            nan_count = X[column].isna().sum()
            if nan_count > 0:
                fill_values = np.random.choice(
                    self.fill_values[column][0], size=nan_count, p=self.fill_values[column][1]
                )
                X.loc[X[column].isna(), column] = fill_values
        return X


# 예: 인코딩 및 스케일링을 포함한 전처리 클래스
class DataPreprocessor:
    __null_columns_proportional = ['income_category']
    __null_columns_simple = ['education_level', 'marital_status']
    __outlier_columns = ["age", "total_trans_cnt"]
    
    def __init__(self):
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.proportional_imputer = ProportionalImputer(columns=self.__null_columns_proportional)
        self.ohe_encoder = ohe_encoder_loaded
        
    def __proportional_impute(self, data):
        self.proportional_imputer.fit(data)
        return self.proportional_imputer.transform(data)
    
    def __simple_impute(self, data):
        data[self.__null_columns_simple] = self.simple_imputer.fit_transform(data[self.__null_columns_simple])
        return data

    def __find_outliers(self, data, column_name, whis=1.5):
        q1, q3 = data[column_name].quantile(q=[0.25, 0.75])
        iqr = q3 - q1
        iqr *= whis
        return data.loc[~data[column_name].between(q1 - iqr, q3 + iqr)]
    
    # Step 1: 결측치 처리
    def __null_feature(self, data):
        self.__proportional_impute(data)
        self.__simple_impute(data)
        
        return data

    # Step 2: 아웃라이어 처리
    def __outlier_feature(self, data, whis=1.5):
        index_list = []
        _data = data.copy()
        
        for col in self.__outlier_columns: 
            outliers_column_index = self.__find_outliers(data, col, whis=whis)
            index_list.extend(outliers_column_index.index)
            
        _data = _data.drop(index=index_list)
            
        _data.reset_index(drop=True, inplace=True)
        
        return _data

    # Step 3: 인코딩
    def __encode_features(self, data):
        # 1. 라벨 인코딩(Label Encoding) - 'gender'
        label_encoder = LabelEncoder()
        data['gender'] = label_encoder.fit_transform(data['gender'])

        # 2. 순서 인코딩 (Ordinal Encoding) - 'education_level', 'income_category'
        education_order = {"Uneducated": 0, "High School": 1, "College": 2, "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5}
        data['education_level'] = data['education_level'].map(education_order)
        income_order = {"Less than $40K" : 0, "$40K - $60K" : 1, "$60K - $80K" : 2,"$80K - $120K" :3, "$120K +":4}
        data['income_category'] = data['income_category'].map(income_order)

        # 4. 원핫 인코딩(One-Hot encoding) - 'marital_status', 'card_category'
        columns_to_ohe_encode = ['marital_status', 'card_category']
        encoded_data_new = self.ohe_encoder.transform(data[columns_to_ohe_encode])
        encoded_df_new = pd.DataFrame(encoded_data_new, columns=self.ohe_encoder.get_feature_names_out())
        data = data.drop(columns=columns_to_ohe_encode)
        
        data = pd.concat([data, encoded_df_new], axis=1)

        return data
    
    def preprocess(self, data):
        data = self.__null_feature(data)
        data = self.__outlier_feature(data)
        data = self.__encode_features(data)
        
        return data