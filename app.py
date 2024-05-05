# 以下を「app.py」に書き込み
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

# 入力フォームを作成
st.title('Leak Rate Prediction')

import joblib
import os

# 保存したモデルを読み込む
model_path = os.path.join(os.path.dirname(__file__), 'lightgbm_model (1).pkl')
bst = joblib.load(model_path)


# 入力フォームでの特徴量の入力
laparotomy = st.selectbox('History of Laparotomy', ['Yes', 'No'])
cT_cat = st.selectbox('cT Category', ['T1/T2', 'T3', 'T4'])
pre_Tx = st.selectbox('Preoperative therapy', ['CRT', 'CT', 'RT', 'none'])
procedure = st.selectbox('Procedure', ['APR', 'Hartmann', 'ISR', 'LAR', 'TPE'])
LPND_lateral = st.selectbox('Lateral PNLD', ['両側', '無', '片側'])
age = st.slider('Age', 0, 100, 50)
bmi = st.slider('BMI', 0.0, 50.0, 25.0, step=0.1)
distance_AV_i = st.slider('Distance from AV', 0.0, 10.0, 0.0,step=0.1)


# ユーザーの入力をデータフレームに変換
input_data = pd.DataFrame({
    'laparotomy': [1 if laparotomy == 'Yes' else 0],
    'cT_cat': {'T1/T2': 0, 'T3': 1, 'T4': 2}[cT_cat],
    'pre_Tx': {'CRT': 0, 'CT': 1, 'RT': 2, 'none': 3}[pre_Tx],
    'procedure': {'APR': 0, 'Hartmann': 1, 'ISR': 2, 'LAR': 3, 'TPE': 4}[procedure],
    'LPND_lateral': {'両側': 0, '無': 1, '片側': 2}[LPND_lateral],
    'age': [age],
    'bmi': [bmi],
    'distance_AV_i': [distance_AV_i]
})

# カテゴリ型に変換する列のリスト
categorical_cols = ['laparotomy', 'cT_cat', 'pre_Tx', 'procedure', 'LPND_lateral']

# 特定の列をカテゴリ型に変換
input_data[categorical_cols] = input_data[categorical_cols].astype('category')

# モデルを使用して「leak」率を予測
prediction = bst.predict(input_data)

# モデルを使用して「leak」率を予測
predicted_leak_rate = prediction[0]


# ボタンを追加
if st.button('Calculate'):
    # モデルを使用して「leak」率を予測
    prediction = bst.predict(input_data)

    # モデルを使用して「leak」率を予測
    predicted_leak_rate = prediction[0]
    predicted_leak_rate_rounded = round(predicted_leak_rate * 100, 1)

    # 結果を表示
    st.write(f'Predicted leak rate: {predicted_leak_rate_rounded:.1f}%')
