import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import _pickle as pickle

model_config = {
    'Model full': {
        'feature_names': ['Age','Sex','Residence','WC','BMI','EL','EF','DP','SS','AI',
             'HR','SBP', 'DBP', 'Hb', 'WBC', 'ALT','AST',
               'SCr', 'TC', 'HDL-C','LDL-C',  
               'HS','T2DM','FHTN'],
        'categorical_features': ['Sex',  'Residence','EL', 'EF','DP', 'SS', 'AI', 'HS','T2DM','FHTN']
    }
}

category_mappings = {
    'HS': {'No': 0, 'Yes': 1},
    'FHTN': {'No': 0, 'Yes': 1},
    'Residence': {'Rural': 1, 'Urban': 2},
    'Sex': {'Male': 1, 'Female': 2},
    'EL': {'Illiterate or semi-literate': 1, "Primary school": 2, "Junior middle school": 3, "Senior middle school": 4, 'College degree and above': 5},
    'EF': {"Never": 1, "Occasionally": 2, "Often": 3},
    'DP': {"Meat and vegetable balance": 1, "Meat based": 2, "Vegetarian based": 3},
    'SS': {"Never": 1, "Smoking": 2, "Quit smoking": 3},
    'AI': {"Never": 1, "Occasionally": 2, "Often": 3},
    'T2DM': {'No': 0, 'Yes': 1},
}

continuous_features_unit_mappings={
    'PLT':'_{(10^9/L)}', 
    'TC':'_{(mmol/L)}',
    'TG':'_{(mmol/L)}',
    'HDL-C':'_{(mmol/L)}',
    'LDL-C':'_{(mmol/L)}',
    'SBP':'_{(mmHg)}',
    'DBP':'_{(mmHg)}',
    'AST':'_{(U/L)}', 
    'ALT':'_{(U/L)}', 
    'Age':'_{(years)}',
    'BMI':'_{(Kg/m^2)}', 
    'TBIL':'_{(μmol/L)}',
    'WBC':'_{(10^9/L)}',
    'SCr':'_{(μmol/L)}',
    'WC':'_{(cm)}',
    'HR':'_{(bpm)}', 
    'Hb':'_{(g/L)}'
}


with open('continuous_features_dict.pkl', 'rb') as f:
    continuous_features_dict = pickle.load(f)


models = {
    'Model full': 'catboost_regressor.pkl',
}

st.title("Risk Score Prediction")

model_name = 'Model full'


with open(models[model_name], 'rb') as f:
    model = pickle.load(f)
explainer = shap.TreeExplainer(model)

with open('iso_reg.pkl','rb') as f:
    iso_reg = pickle.load(f)

feature_names = model_config[model_name]['feature_names']
categorical_features = model_config[model_name]['categorical_features']
continuous_features = [i for i in feature_names if i not in categorical_features]


col1, col2 = st.columns(2)

with col1:
    st.header("Input Parameters")
    input_data = {}

    for var in categorical_features:
        options = list(category_mappings[var].keys())
        input_data[var] = st.selectbox(f"${var}$", options=options)

    for var in continuous_features:
        second_dict = continuous_features_dict[var]
        range_min = float(int(second_dict['range_min']))
        range_max = float(int(second_dict['range_max']))
        step = second_dict['step']
        continuous_features_unit = continuous_features_unit_mappings[var]
        input_data[var] = st.slider(f"${var} {continuous_features_unit}$",min_value=range_min, max_value=range_max, value=float(int(second_dict['mean'])), step=step)


with col2:
    st.header("Prediction Result and Feature Importance")

    if st.button("Calculate"):
        for var in categorical_features:
            input_data[var] = category_mappings[var][input_data[var]]

        input_AI = pd.DataFrame([input_data])
        input_AI = input_AI.reindex(columns=feature_names)
        for i in categorical_features:
            input_AI[i] = input_AI[i].astype('category')

        st.write("Input DataFrame:")
        st.write(input_AI)

        prediction = model.predict_proba(input_AI)[0, 1]
        prediction = iso_reg.transform([prediction])[0]
        if model_name == 'Model full':
            if prediction < 0.15:
                risk = 'Low'
            elif prediction >= 0.8:
                risk = 'Very high'
            elif 0.3<=prediction < 0.8:
                risk = 'High'
            else:
                risk = 'Medium'

        st.write(f"Predicted Risk Score: {risk}")
        st.write(f"Prediction Value: {np.round(prediction,3)}")

        shap_values = explainer(input_AI)
        feature_importance = shap_values.values[0]
        feature_names.append('Output')
        feature_importance = np.append(feature_importance, explainer.expected_value + feature_importance.sum())
        importance_AI = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_AI['Color'] = importance_AI['Importance'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

        fig = px.bar(importance_AI, x='Feature', y='Importance', color='Color',
                     color_discrete_map={'Positive': 'red', 'Negative': 'blue'},
                     title='Predicted Tips explained by SHAP values')
        st.plotly_chart(fig)
