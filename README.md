# SKN06-2nd-4Team

SKN6기 2nd 단위 프로젝트 - 공인용, 김동명, 박유나, 임연경

## 가디언즈 오브 독산

### 팀원

| 공인용                                                                                                                                                                                                                                                                                                                                                                        | 김동명                                                                                                                                                                                                                                                                                                                    | 박유나                                                                                                                                                                                                                             | 임연경                                                                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxODA0MTFfMTM0%2FMDAxNTIzNDU0NzkwNDE5.gEK35xFgCX9vYBn5oOJyWuh2e5pbuhlSbfdl6poV5uEg.Mv_Dfnxo30cnaT6L6CRO1qnHuhw-w2-IOrbCvdavhJ8g.JPEG.hong5395%2F3e9932eb44c7e5d95e3380f0b3850a10849e64a53fc218b5f937de3f8aa32c7d179cdaa4ff41.jpg&type=sc960_832" alt="image" width="200" height="250"/> | <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxNzAzMDVfMTA2%2FMDAxNDg4Njk1MjMzMzM5.n8IyNuI2bfb9ahCzK5BuXarECmC0kAgwXAQ_VqnVfvkg.YVo_NXVr0Yum_arFadeksjm5EK2llgXS7c5_gdAAyk0g.JPEG.herotime01%2Fmms411-r9_shop1_151309.jpg&type=sc960_832" alt="image" width="200" height="250"/> | <img src="https://search.pstatic.net/sunny/?src=https%3A%2F%2Fi.namu.wiki%2Fi%2FHTd0cQVU-2HsObW-meRcGxERbzgr80e3y0K2IkUPVuAtCAQgoN684suvdC3B3vAr6G_lT_XJk4j5k7l-_7sLbg.webp&type=sc960_832" alt="image" width="200" height="250"/> | <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxODA1MjVfMTMz%2FMDAxNTI3MTk1NDA2NDY4.52sHvf5xhDFsa547Q0XppzIAaz_LuXEm1vIkHcXev8Ag.dxIsd79lNL_5QekTes5_Agf4EzveLb7L1Ub-EHP738Ag.JPEG.loyh%2FDSC01289.JPG&type=a340" alt="image" width="200" height="250"/> |
| ML                                                                                                                                                                                                                                                                                                                                                                            | ML, 발표                                                                                                                                                                                                                                                                                                                  | ML, Streamlit                                                                                                                                                                                                                      | ML                                                                                                                                                                                                                                                                                               |

</br>

# 💳 신용카드 이용 고객 - 이탈 예측 모델 💳

### ✔️ 개발 기간

2024.11.14 ~ 2024.11.15(총 2일)

### ✔️ 개요

금융 서비스 시장에서 고객 이탈 방지는 수익성 유지와 경쟁력 강화를 위해 필수적이다. 신용카드 사용 고객의 이탈 패턴을 이해하고, 효과적인 고객 유지 전략을 세우기 위해 이탈 예측 모델이 필요하다. 이를 통해 금융 기관은 데이터 기반 의사결정을 강화하고, 고객 맞춤형 마케팅 전략을 수립할 수 있다.

### ✔️ 목표

신용카드 사용 고객의 데이터를 분석하여 이탈 가능성을 예측하는 모델을 개발

</br>

#### ✔️ Stacks

![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Numpy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
<img src="https://img.shields.io/badge/scikitlearn-%23F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>

![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
</br>

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)

#### ✔️ Requirements

streamlit == 1.39.0 <br/>
pymysql == 1.1.1 <br/>
pandas == 2.2.3 <br/>
openpyxl == 1.1.0 <br/>
sqlalchemy == 2.0.35 <br/>
configparser == 7.1.0 <br/>
matplotlib == 3.9.2 <br/>
xlrd == 2.0.1 <br/>
seaborn == 0.13.2 <br/>
joblib == 1.3.2 <br/>
scikit-learn == 1.3.1 <br/>
numpy == 1.26.0 <br/>
xgboost == 1.7.6 <br/>

## 데이터 준비 및 분석

### ✔️ Column 정의

[Google 스프레드시트 보기](https://docs.google.com/spreadsheets/d/1PvMto9SCOenoNsXg_mjzhMyAeArdpVEP5e5ZlOuftFI/edit?usp=sharing)

| Column 이름             | Description                | Feature Value                                                                               | 비고                                                                                                                                                    |
| ----------------------- | -------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CLIENTNUM               | 고객 번호                  | n                                                                                           | 삭제 필요                                                                                                                                               |
| churn                   | 결과 값(churn)             | "Existing", "Attrited"                                                                      | - Encoding → Label Encoding                                                                                                                             |
| age                     | 나이                       | 26~57                                                                                       |                                                                                                                                                         |
| gender                  | 성별                       | M, F                                                                                        | - Encoding → Label Encoding                                                                                                                             |
| dependent_cnt           | 부양 가족수                | 0, 1, 2, 3, 4, 5                                                                            |                                                                                                                                                         |
| education_level         | 학력                       | "Graduate", "High School", "Unknown", "Uneducated", "College", "Post-Graduate", "Doctorate" | - 결측치 처리 필요: Unknown(1519개/0.149) → 최빈값 <br> - graduate의 비율이 높음<br>- Encoding - 순서의 의미가 있어 보임 → 순서 인코딩 (Ordinal Encoding)|
| marital_status          | 결혼 여부                  | "Married", "Single", "Unknown", "Divorced"                                                  | - 결측치 처리 필요: Unknown(749개/0.07) → 최빈값<br>- married의 비율이 높음<br>- Encoding → One-Hot Encoding                                            |
| income_category         | 소득 수준(범주)            | "Unknown", "Less than $40K", "$40K - $60K", "$80K - $120K", "$60K - $80K", "$120K +"        | - 결측치 처리 필요: Unknown(1112개/0.109) → 비례배분<br>- Encoding - 순서의 의미가 있어 보임 → 순서 인코딩 (Ordinal Encoding)                           |
| card_category           | 카드 종류(범주)            | "Blue", "Silver", "Gold", "Platinum"                                                        | - Encoding → One-Hot Encoding                                                                                                                           |
| card_usage_period       | 카드 사용 기간             | n                                                                                           | - max : 56 > min : 13<br>- 0.15 기준으로 탈락                                                                                                           |
| account_cnt             | 계좌 수                    | 1, 2, 3, 4, 5, 6                                                                            |                                                                                                                                                         |
| inactive_month_in_year  | 연내 계좌 비활성 기간      | 0, 1, 2, 3, 4, 5, 6                                                                         |                                                                                                                                                         |
| visit_cnt_in_year       | 연간 은행 방문 수          | 0, 1, 2, 3, 4, 5, 6                                                                         |                                                                                                                                                         |
| credit_limit            | 신용 한도                  | n                                                                                           | - max: 34516.0 > min: 1438.3                                                                                                                            |
| revolving_balance       | 잔금                       | n                                                                                           | - max: 2517 > min: 1438.3                                                                                                                               |
| avg_remain_credit_limit | 평균 잔여 신용 한도        | n                                                                                           | - max : 34516.0 > min: 3.0                                                                                                                        |
| total_amt_change_Q4_Q1  | 연간 거래액 변화율(Q4/Q1)  | n                                                                                           | - max : 3.397 / min: 0<br>                                                                                                                              |
| total_trans_amt         | 총 거래 금액               | n                                                                                           | - max : 18484 / min: 510                                                                                                                                |
| total_trans_cnt         | 총 거래 횟수               | n                                                                                           | - max : 139 / min : 0<br>                                                                                                                               |
| total_cnt_change_Q4_Q1  | 총 거래 횟수 변화율(Q4/Q1) | n                                                                                           |                                                                                                                                                         |
| avg_utilization_ratio   | 카드 한도 대비 잔액의 비율 | 0 <= n <= 1                                                                                 | - 0 ~ 1 실수<br>- '0' 의 비율이 높음                                                                                                                    |

### ✔️ EDA(탐색적 데이터 분석)

![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/EDA.png)

<br/>

## 데이터 전처리

### ✔️ 1. 결측치 처리

⭐️ 3개의 문자열 칼럼에서 'Unknown' 결측치가 발견됐다. 다양한 처리 방법 중 삭제를 고려하기도 했지만, 삭제할 경우 데이터 손실이 많아질 것 같아 **대체** 방법을 선택했다.

| education_level                                                                                                                             | marital_status                                                                                                                              | income_category                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%ED%95%99%EB%B2%8C.png) | ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%EA%B2%B0%ED%98%BC.png) | ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%EC%9E%90%EC%82%B0.png) |
| SimpleImputer(**최빈값**)                                                                                                                   | SimpleImputer(**최빈값**)                                                                                                                   | 사용자 정의 imputer(**가중대체**)                                                                                                           |
| unkown의 비율이 나머지에 비에 높지 않음                                                                                                     | unkown의 비율이 나머지에 비에 높지 않음                                                                                                     | unkown의 비율이 나머지에 비에 높음                                                                                                          |
| Graduate가 가장 많은 비율(30.89%)을 차지                                                                                                    | Married가 가장 높은 비율(46.28%)을 차지                                                                                                     | 각각 나머지 자료의 비율에 따라 랜덤으로 분배                                                                                                |

</br>
👉🏻 우리가 '<b>가중대체</b>'를 위해 정의한 Imputer
</br>
</br>

```python
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
```

### ✔️ 2. 이상치 처리

![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/boxplot.png)
</br>
⭐️ IQR을 사용해 이상치를 확인한 결과, 일부 이상치로 추정되는 데이터를 발견했다. 그중 ["age", "total_trans_cnt"] 칼럼에서 각각 2개의 극단치 데이터가 결과에 거의 영향을 미치지 않을 것으로 판단되어 삭제하기로 했다.
</br>
</br>

```python
def __outlier_feature(self, data, whis=1.5):
        index_list = []
        _data = data.copy()

        for col in self.__outlier_columns:
            outliers_column_index = self.__find_outliers(data, col, whis=whis)
            index_list.extend(outliers_column_index.index)

        _data = _data.drop(index=index_list)

        _data.reset_index(drop=True, inplace=True)

        return _data
```

### ✔️ 3. Feature Engineering

데이터 특징별 인코딩 방식은 아래와 같다. - 4가지 방법으로 적용

1. 라벨 인코딩(Label Encoding)
   > 'gender'
   >
   > 이진 변수의 경우 모델 성능에 큰 차이가 없으므로, 간단히 라벨 인코딩을 사용하기로 함.
2. 순서 인코딩 (Ordinal Encoding)
   > 'education_level', 'income_category'
   >
   > 학력과 소득과 관련된 자료는 자료량이 아닌 해당 index로 순서를 결정하기 위함.
3. mapping
   > 'churn'
   >
   > 이탈한 고객을 1로 설정하고 이탈하지 않은 고객을 0으로 설정해 자료의 분석을 쉽게할 수 있도록 함.
4. 원핫 인코딩(One-Hot encoding)
   > 'marital_status', 'card_category'
   >
   > 순서가 없고 각 값이 독립적인 범주형 데이터으로서 순서나 크기 정보 없이 각각 독립적인 특성으로 변환되므로, 머신러닝 모델에서 더 잘 해석될 가능성이 있다고 보아 OneHot 인코딩 하기로 결정.
   > models/ohe_encoder.pkl 로 저장

<br/>

```python
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
```

### 📌 전처리 Preprocessor 정의

- 위의 전처리 과정들을 실행시키는 클래스 정의 → DataPreprocessor

```python
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

```

## 모델링

### ✔️ 모델 선정하기

데이터와 어울리는 7개의 모델들은 뽑아 어떤 모델이 적합할지 확인해 보기로 했다.

- 평가

```
  from tqdm import tqdm

  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import GradientBoostingClassifier
  from xgboost import XGBClassifier, plot_importance
  from sklearn.svm import SVC
  from sklearn.neighbors import KNeighborsClassifier

  import matplotlib.pyplot as plt
  models = {
      # Logistic Regression model
      "Logistic Regression": LogisticRegression(),
      # Decision Tree model
      "Decision Tree Classifier": DecisionTreeClassifier(),
      # Random Forest model
      "Random Forest": RandomForestClassifier(),
      # Gradient Boosting model
      "Gradient Boosting": GradientBoostingClassifier(),
      # XGBoost model
      "XGBoost": XGBClassifier(),
      # SVM(Support Vector Machine)
      "SVC": SVC(),
      # KNN(K-Nearest Neighbors)
      "KNeighborsClassifier": KNeighborsClassifier(),
  }


  for name, model in tqdm(models.items(), desc="Training Models", total=len(models)):
      # 모델 훈련
      model.fit(X_train, y_train)
      # 모델 평가
      score = model.score(X_test, y_test)
      # 모델 검증
      model_pred = model.predict(X_test)
      # 모델 정확도
      tqdm.write(f">>> {name} : 정확도 {score:.2%}\n")

```

- 결과

```python
>>> Logistic Regression : 정확도 87.90%

>>> Decision Tree Classifier : 정확도 94.02%

>>> Random Forest : 정확도 95.65%

>>> Gradient Boosting : 정확도 96.15%

>>> XGBoost : 정확도 96.74%

>>> SVC : 정확도 84.20%

>>> KNeighborsClassifier : 정확도 90.47%
```

#### ⭐ 선정 결과

- LogisticRegression
- DecisionTreeClassifier (✔️) - 김동명
- RandomForestClassifier (✔️) - 임연경
- GradientBoostingClassifier (✔️) - 박유나
- xgboost (✔️) - 공인용
- SVC
- KNeighborsClassifier

7개의 모델 중 4개의 모델이 우수한 편이었고, 각자 모델 한개씩 맡아서 모델링을 하기로 했다.

### ✔️ 머신 러닝 모델

#### 1. Decision Tree Classifier : 정확도 93.78%

- 주요 파라미터

  > criterion: 노드 분할 기준
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > min_samples_split: 노드를 분할하기 위한 최소 샘플 수
  >
  > min_samples_leaf: 리프 노드의 최소 샘플 수
  >
  > max_features: 각 트리가 학습할 때마다 사용할 특성(feature)의 수

  ```

  from sklearn.tree import DecisionTreeClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



  tree = DecisionTreeClassifier()

  tree.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_tree = tree.predict(X_train)
  y_train_proba_tree= tree.predict_proba(X_train)[:, 1]

  y_test_pred_tree = tree.predict(X_test)
  y_test_proba_tree= tree.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_tree)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Decision Tree (Test Set)")
  plt.show()

  evaluate("Train - Decision Tree", y_train, y_train_pred_tree, y_train_proba_tree)
  evaluate("Test - Decision Tree", y_test, y_test_pred_tree, y_test_proba_tree)

  # 3. 특성 중요도 계산 및 시각화
  fi = tree.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Decision Tree")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      'criterion': ['gini', 'entropy'],  # 노드 분할 기준
      'max_depth': [None, 10, 20, 30],   # 각 결정 트리의 최대 깊이를 설정
      'min_samples_split': [2, 10, 20],  # 노드를 분할하기 위한 최소 샘플 수
      'min_samples_leaf': [1, 5, 10],    # 리프 노드의 최소 샘플 수
      'max_features': [None, 'sqrt', 'log2']  # 각 트리가 학습할 때마다 사용할 특성(feature)의 수
  }

  gs_tree = GridSearchCV(
      estimator=tree,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_tree.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_tree = gs_tree.best_params_
  best_model_tree = gs_tree.best_estimator_

  best_y_pred_tree = best_model_tree.predict(X_test)
  best_y_proba_tree= best_model_tree.predict_proba(X_test)[:, 1]

  ```

#### 2. Random Forest : 정확도 95.65%

- 주요 파라미터

  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > max_features: 각 트리가 학습할 때마다 사용할 특성(feature)의 수

  ```
  from sklearn.ensemble import RandomForestClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  rf = RandomForestClassifier()

  rf.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_rf = rf.predict(X_train)
  y_train_proba_rf= rf.predict_proba(X_train)[:, 1]

  y_test_pred_rf = rf.predict(X_test)
  y_test_proba_rf= rf.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_rf)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Random Forest (Test Set)")
  plt.show()

  evaluate("Train - Random Forest", y_train, y_train_pred_rf, y_train_proba_rf)
  evaluate("Test - Random Forest", y_test, y_test_pred_rf, y_test_proba_rf)

  # 3. 특성 중요도 계산 및 시각화
  fi = rf.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Random Forest")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      'n_estimators': [100, 200, 300],    # 결정 트리(Decision Tree)의 개수
      'max_depth': [5, 10, 15],           # 각 결정 트리의 최대 깊이를 설정
      'max_features': ['sqrt', 'log2']    # 각 트리가 학습할 때마다 사용할 특성(feature)의 수
  }
  gs_rf = GridSearchCV(
      estimator=rf,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_rf.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_rf = gs_rf.best_params_
  best_model_rf = gs_rf.best_estimator_

  best_y_pred_rf = best_model_rf.predict(X_test)
  best_y_proba_rf= best_model_rf.predict_proba(X_test)[:, 1]

  ```

#### 3. Gradient Boosting : 정확도 96.79%

- 주요 파라미터

  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > learning_rate: 학습률
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > subsample: 각 트리 학습에 사용되는 샘플의 비율

  ```
  from sklearn.ensemble import GradientBoostingClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  gb = GradientBoostingClassifier()

  gb.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_gb = gb.predict(X_train)
  y_train_proba_gb= gb.predict_proba(X_train)[:, 1]

  y_test_pred_gb = gb.predict(X_test)
  y_test_proba_gb= gb.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_gb)
  plt.figure(figsize=(6,4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Gradient Boosting (Test Set)")
  plt.show()

  evaluate("Train - Gradient Booting", y_train, y_train_pred_gb, y_train_proba_gb)
  evaluate("Test - Gradient Booting", y_test, y_test_pred_gb, y_test_proba_gb)

  # 3. 특성 중요도 계산 및 시각화
  fi = gb.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Gradient Boosting")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      "n_estimators": [100, 200, 300],  #  부스팅 단계의 수 = 모델이 생성할 트리 개수
      "learning_rate": [0.1],  # 학습률
      "max_depth": [1, 2, 3, 4, 5],  # 각 결정 트리의 최대 깊이를 설정
      "subsample": [0.5, 0.7],  # 샘플링 비율
  }

  gs_gb = GridSearchCV(
      estimator=gb,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_gb.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_gb = gs_gb.best_params_
  best_model_gb = gs_gb.best_estimator_

  best_y_pred_gb = best_model_gb.predict(X_test)
  best_y_proba_gb= best_model_gb.predict_proba(X_test)[:, 1]

  ```

#### 4. XGBoost : 정확도 97.19%

- 주요 파라미터

  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > learning_rate: 학습률
  >
  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > subsample: 각 트리의 훈련에 사용되는 샘플 비율
  >
  > colsample_bytree: 각 트리의 훈련에 사용되는 피처 비율
  >
  > gamma: 노드 분할에 대한 최소 손실 감소
  >
  > reg_alpha: L1 정규화
  >
  > reg_lambda: L2 정규화

  ```
  from xgboost import XGBClassifier

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  xgb = XGBClassifier()

  xgb.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_xgb = xgb.predict(X_train)
  y_train_proba_xgb= xgb.predict_proba(X_train)[:, 1]

  y_test_pred_xgb = xgb.predict(X_test)
  y_test_proba_xgb= xgb.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_xgb)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - XGBoost (Test Set)")
  plt.show()

  evaluate("Train - XGBoost", y_train, y_train_pred_xgb, y_train_proba_xgb)
  evaluate("Test - XGBoost", y_test, y_test_pred_xgb, y_test_proba_xgb)

  # 3. 특성 중요도 계산 및 시각화
  fi = xgb.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in XGBoost")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      "max_depth":[1, 2, 3, 4, 5],            # 각 결정 트리의 최대 깊이를 설정
      'learning_rate': [0.1],                 # 학습률
      'n_estimators': [100, 200, 300],        # 부스팅 단계의 수 = 모델이 생성할 트리 개수
      'subsample': [0.5, 0.7],                # 각 트리의 훈련에 사용되는 샘플 비율
      'colsample_bytree': [0.5, 0.7, 1.0],    # 각 트리의 훈련에 사용되는 피처 비율
      'gamma': [0, 0.1],                      # 노드 분할에 대한 최소 손실 감소
      'reg_alpha': [0],                       # L1 정규화
      'reg_lambda': [0.1]                     # L2 정규화
  }
  gs_xgb = GridSearchCV(
      estimator=xgb,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_xgb.fit(X_train, y_train)

  # 5. 튜닝 : Best Model 찾기
  best_param_xgb = gs_xgb.best_params_
  best_model_xgb = gs_xgb.best_estimator_

  best_y_pred_xgb = best_model_xgb.predict(X_test)
  best_y_proba_xgb= best_model_xgb.predict_proba(X_test)[:, 1]

  ```

| 머신러닝 방법    | Decision Tree Classifier                                                                                                                                                                                                                | Random Forest                                                                                                                                                                                                                                             | Gradient Boosting                                                                                                                                                                                           | XGBoost                                                                                                                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Confusion Matrix | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-cm.png" alt="image" width="200" height="200"/>                                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-cm.png" width="200" height="200"/>                                                                          | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-cm.png" alt="image" width="200" height="200"/>                                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-cm.png" alt="image" width="200" height="200"/>                                                              |
| 결과             | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%EA%B2%B0%EA%B3%BC.png" width="300" height="150"/>                                                          | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              |
| 특성중요도       | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" width="300" height="150"/>                               | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   |
| 하이퍼파라미터   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="160"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="100"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGBoost-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> |

### ✔️ 모델 평가

```
# 여러 평가 지표 설정
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'auc': make_scorer(roc_auc_score)
}

model_box = pd.DataFrame(columns=['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost'],
                            index = ['accuracy','precision','recall','f1 score','auc'])

def evaluate(title, y_real, y_pred, y_prob):
    acc = accuracy_score(y_real, y_pred)
    pre = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_prob)

    print(f"======= {title} =======")
    print('Accuracy : {:.6f}'.format(acc)) # 정확도 : 예측이 정답과 얼마나 정확한가
    print('Precision : {:.6f}'.format(pre)) # 정밀도 : 예측한 것 중에서 정답의 비율
    print('Recall : {:.6f}'.format(rec)) # 재현율 : 정답 중에서 예측한 것의 비율
    print('F1 score : {:.6f}'.format(f1)) # 정밀도와 재현율의 (조화)평균 - 정밀도와 재현율이 비슷할수록 높은 점수
    print('auc: {:.6f}'.format(auc))


    score_list = [acc,pre,rec,f1,auc]
    score_box = np.array(score_list)

    return score_box
```

![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EC%A0%95%ED%99%95%EB%8F%84.png)

### ✔️ 최고 성능 모델

🏆 XGBOOST

<br/>

## 모델 저장

하이퍼파라미터 튜닝을 통해 각 모델별 best params 를 통해 만든 best model들을 .pkl 파일로 저장.

```
import os
import joblib

directory = 'model/'
os.makedirs(directory, exist_ok=True)

joblib.dump(best_model_tree, os.path.join(directory, 'best_tree.pkl'))
joblib.dump(best_model_rf, os.path.join(directory, 'best_rf.pkl'))
joblib.dump(best_model_gb, os.path.join(directory, 'best_gb.pkl'))
joblib.dump(best_model_xgb, os.path.join(directory, 'best_xgb.pkl'))

# 저장된 모델과 파라미터 불러오기
model_tree = joblib.load('model/best_tree.pkl')
model_rf = joblib.load('model/best_rf.pkl')
model_gb = joblib.load('model/best_gb.pkl')
model_xgb = joblib.load('model/best_xgb.pkl')
```

## Streamlit

