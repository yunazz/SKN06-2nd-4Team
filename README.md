# SKN06-2nd-4Team
SKN6기 2nd 단위 프로젝트 - 공인용, 김동명, 박유나, 임연경
## 가디언즈 오브 독산
<img src="https://postfiles.pstatic.net/20140724_24/sound_tel_1406201694894NzGTt_PNG/Screenshot_2014-07-24-20-25-03_edit.png?type=w1" alt="image" width="200" height="300"/>

### 팀원
| 공인용 | 김동명 | 박유나 | 임연경 |
|--|--|--|--|
| <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxODA0MTFfMTM0%2FMDAxNTIzNDU0NzkwNDE5.gEK35xFgCX9vYBn5oOJyWuh2e5pbuhlSbfdl6poV5uEg.Mv_Dfnxo30cnaT6L6CRO1qnHuhw-w2-IOrbCvdavhJ8g.JPEG.hong5395%2F3e9932eb44c7e5d95e3380f0b3850a10849e64a53fc218b5f937de3f8aa32c7d179cdaa4ff41.jpg&type=sc960_832" alt="image" width="200" height="250"/>| <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxNzAzMDVfMTA2%2FMDAxNDg4Njk1MjMzMzM5.n8IyNuI2bfb9ahCzK5BuXarECmC0kAgwXAQ_VqnVfvkg.YVo_NXVr0Yum_arFadeksjm5EK2llgXS7c5_gdAAyk0g.JPEG.herotime01%2Fmms411-r9_shop1_151309.jpg&type=sc960_832" alt="image" width="200" height="250"/>| <img src="https://search.pstatic.net/sunny/?src=https%3A%2F%2Fi.namu.wiki%2Fi%2FHTd0cQVU-2HsObW-meRcGxERbzgr80e3y0K2IkUPVuAtCAQgoN684suvdC3B3vAr6G_lT_XJk4j5k7l-_7sLbg.webp&type=sc960_832" alt="image" width="200" height="250"/>|<img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxODA1MjVfMTMz%2FMDAxNTI3MTk1NDA2NDY4.52sHvf5xhDFsa547Q0XppzIAaz_LuXEm1vIkHcXev8Ag.dxIsd79lNL_5QekTes5_Agf4EzveLb7L1Ub-EHP738Ag.JPEG.loyh%2FDSC01289.JPG&type=a340" alt="image" width="200" height="250"/>|

</br></br></br>

## 💳 신용카드 이용 고객데이터 분석 및 이탈 예측 모델 💳

### ✔️ 개발 기간
2024.11.14~2024.11.15(총 2일)

### ✔️ 프로젝트 필요성 (배경)
금융 서비스 시장에서 고객 이탈 방지는 수익성 유지와 경쟁력 강화를 위해 필수적이다. 신용카드 사용 고객의 이탈 패턴을 이해하고, 효과적인 고객 유지 전략을 세우기 위해 이탈 예측 모델이 필요하다. 이를 통해 금융 기관은 데이터 기반 의사결정을 강화하고, 고객 맞춤형 마케팅 전략을 수립할 수 있다.

### ✔️ 프로젝트 목표
신용카드 사용 고객의 데이터를 분석하여 이탈 가능성을 예측하는 모델을 개발하고, 이탈에 영향을 미치는 원인을 알아냄으로써 경제 시장에서의 고객 관리 전략을 개선할 수 있다.


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

streamlit == 1.39.0
pymysql == 1.1.1
pandas == 2.2.3
openpyxl == 1.1.0
sqlalchemy == 2.0.35
configparser == 7.1.0
matplotlib == 3.9.2
xlrd == 2.0.1
seaborn == 0.13.2
joblib == 1.3.2
scikit-learn == 1.3.1
numpy == 1.26.0
xgboost == 1.7.6


## 데이터 전처리
### ✔️ 변수 정의
[Google 스프레드시트 보기](https://docs.google.com/spreadsheets/d/1PvMto9SCOenoNsXg_mjzhMyAeArdpVEP5e5ZlOuftFI/edit?usp=sharing)

### ✔️ EDA(탐색적 데이터 분석)
![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/EDA.png)
### ✔️ 결측치 처리
⭐️ 문자열 형식으로 된 education_level, marital_status, income_category에서 Unknown이라는 결측치 발생 ⭐️
| education_level | marital_status | income_category |
|--|--|--|
| ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%ED%95%99%EB%B2%8C.png) | ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%EA%B2%B0%ED%98%BC.png) | ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%B8%A1%EC%B9%98%20%EC%9E%90%EC%82%B0.png) |
| SimpleImputer(최빈값) | SimpleImputer(최빈값) | 사용자 정의 imputer(가중대체) |
| unkown의 비율이 나머지에 비에 높지 않음 | unkown의 비율이 나머지에 비에 높지 않음 | unkown의 비율이 나머지에 비에 높음 |
| Graduate가 가장 많은 비율(30.89%)을 차지 | Married가 가장 높은 비율(46.28%)을 차지 | 각각 나머지 자료의 비율에 따라 랜덤으로 분배 |
</br>
👉🏻 우리가 정의한 imputer
</br>



```
 Class ProportionalImputer(BaseEstimator, TransformerMixin):
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




### ✔️ 이상치 처리

![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/boxplot.png)


```
def find_outliers(df, column_name, whis=1.5):
    q1, q3 = df[column_name].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    iqr *= whis
    return df.loc[~df[column_name].between(q1 - iqr, q3 + iqr)]
```




</br>
⭐️ 발생한 여러 이상치들 중 결과에 영향을 거의 미치지 않을 것같은 이상치 삭제 => ["age", "total_trans_cnt"] 두 칼럼의 이상치를 삭제하기로 결정 ⭐️
</br>





```
def delete_outliers(df, columns, whis=1.5):
    index_list = []
    _df = df.copy()
    
    for col in columns: 
        outliers_column_index = find_outliers(df, col, whis=whis)
        index_list.extend(outliers_column_index.index)
        
        
    _df = _df.drop(index=index_list)
        
    _df.reset_index(drop=True, inplace=True)
    
    return _df

outlier_columns = ["age", "total_trans_cnt"]
df = delete_outliers(df, outlier_columns)
```





### ✔️ Feature Engineering
⭐️ 문자열 자료들을 숫자형으로 변경하기 위해 진행 ⭐️
1. 라벨 인코딩(Label Encoding) 
> 'gender'
> 
> 이진 변수의 경우 모델 성능에 큰 차이가 없으므로, 간단히 라벨 인코딩을 사용하기로 함.
> 
2. 순서 인코딩 (Ordinal Encoding)
> 'education_level', 'income_category'
>
> 학력과 소득과 관련된 자료는 자료량이 아닌 해당 index로 순서를 결정하기 위함.
> 
3. mapping
> 'churn'
>
> 이탈한 고객을 1로 설정하고 이탈하지 않은 고객을 0으로 설정해 자료의 분석을 쉽게할 수 있도록 함.
> 
4. 원핫 인코딩(One-Hot encoding)
> 'marital_status', 'card_category'
> 
> 순서가 없고 각 값이 독립적인 범주형 데이터으로서 순서나 크기 정보 없이 각각 독립적인 특성으로 변환되므로, 머신러닝 모델에서 더 잘 해석될 가능성이 있다고 보아 OneHot 인코딩 하기로 결정.
>

## 모델 학습 결과서
### 모델 평가에 사용된 평가 지표
- 'id', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2' 을 제외한 다른 지표
> 'id' : 모든 행들이 가지고 있는 고유의 값으로 평가에는 도움이 되지 않음.
> 
> 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2' : 이 지표들은 이미 기존 모델에서 계산된 확률 값이므로, 새 모델에 포함시키면 편향이나 과적합을 유발할 수 있음.


### 과정
- 전체적으로 어떤 모델이 적합할지 확인
  ![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20%EC%B2%AB%EA%B2%B0%EA%B3%BC.png)
- 우수 모델 4가지를 선택해 파라미터 설정 등 자세한 분석 시행
  > Decision Tree Classifier : 정확도 93.78%
  >
  > Random Forest : 정확도 95.65%
  >
  > Gradient Boosting : 정확도 96.79%
  >
  > XGBoost : 정확도 97.19%
  >
|머신러닝 방법| Decision Tree Classifier | Random Forest | Gradient Boosting | XGBoost |
|--|--|--|--|--|
|Confusion Matrix| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-cm.png" alt="image" width="200" height="200"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-cm.png" width="200" height="200"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-cm.png" alt="image" width="200" height="200"/>|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-cm.png" alt="image" width="200" height="200"/>|
|결과| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%EA%B2%B0%EA%B3%BC.png" width="300" height="150"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>|
|특성중요도| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" width="300" height="150"/>| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>|
|하이퍼파라미터| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="160"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="100"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGBoost-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> |

</br>

![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EC%A0%95%ED%99%95%EB%8F%84.png)

### 최종 선정 모델
⭐️ XGBoost ⭐️
```
import os
import joblib

# 최적 모델과 하이퍼파라미터 추출
best_models = {
    'DecisionTreeClassifier': best_model_tree,
    'RandomForestClassifier': best_model_rf,
    'GradientBoostingClassifier': best_model_gb,
    'XGBClassifier': best_model_xgb
}

params = {
    'DecisionTreeClassifier': best_model_tree,
    'RandomForestClassifier': best_model_rf,
    'GradientBoostingClassifier': best_model_gb,
    'XGBClassifier': best_model_xgb
}

# 모델과 하이퍼파라미터를 하나의 딕셔너리로 묶어서 저장
model_dict = {
    'models': best_models,
    'params': params
}

import joblib

directory = 'saved/'
os.makedirs(directory, exist_ok=True)

joblib.dump(model_dict, os.path.join(directory, 'models_and_params.joblib'))

# 저장된 모델과 파라미터 불러오기
loaded_model_dict = joblib.load('/saved/models_and_params.joblib')

# 모델과 하이퍼파라미터 출력
print("Loaded Models:", loaded_model_dict['models'])
print("Loaded Params:", loaded_model_dict['params'])
```

### streamlit 결과

## ✔️ 팀원 회고

공인용
> 
> 
김동명
> 
> 
박유나
> 
>
임연경

### 추후 계선 사항
- readme.md 상세 작성
- 전처리 pipline
- 예측 후 평가
  
'colsample_bytree': 1.0 
