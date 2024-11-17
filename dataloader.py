import pandas as pd


def load_dataset():
    # 데이터 load
    data = pd.read_csv("data/credit_card_churn.csv", na_values="Unknown")

    # 컬럼명 변경
    rename_columns = {
        "Attrition_Flag": "churn",
        "Customer_Age": "age",
        "Dependent_count": "dependent_cnt",
        "Months_on_book": "card_usage_period",
        "Total_Relationship_Count": "account_cnt",
        "Months_Inactive_12_mon": "inactive_month_in_year",
        "Contacts_Count_12_mon": "visit_cnt_in_year",
        "Total_Revolving_Bal": "revolving_balance",
        "Avg_Open_To_Buy": "avg_remain_credit_limit",
        "Total_Amt_Chng_Q4_Q1": "total_amt_change_q4_q1",
        "Total_Trans_Ct": "total_trans_cnt",
        "Total_Ct_Chng_Q4_Q1": "total_cnt_change_q4_q1",
    }
    data.rename(columns=rename_columns, inplace=True)
    # 컬럼명 소문자로 변경
    data.columns = data.columns.str.lower()

    ## 컬럼 삭제
    data.drop(
        columns=[
            "clientnum",
            "naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1",
            "naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2",
        ],
        inplace=True,
    )

    X = data.drop(columns="churn")
    y = data["churn"]

    return X, y
