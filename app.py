import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder

#  streamlit
st.title('Disease Prediction system')
st.subheader("Hey there 👋 You are Most Welcome 😊!!")
st.write("If you want to forcast the Disease that You may be suffering from then Yes!! You are at the Right Place✅. Go ahead. You will definately find this Usefull !!")

# dataset view
df_train=pd.read_csv('train_data.csv')
st.write("Dataset preview (Only first 5 rows are displayed here):",df_train.head(5))

# Encode categorical variable
for column in df_train.columns[:-1]:
    leb_encoder=LabelEncoder()
    df_train[column]=leb_encoder.fit_transform(df_train[column])


#  define Features and target varible
X_train=df_train.drop(columns=['disease'])
y_train=df_train['disease']


# Train Decision Tree model
Decision_Tree_model = DecisionTreeClassifier()
Decision_Tree_model.fit(X_train, y_train)

# Extract and views all Diseases that are available
st.subheader('take a look at all available Diseases that can be predicted:')
unique_diseases=df_train['disease'].unique()
num_cols=5
cols=st.columns(num_cols)
for idx,disease in enumerate(unique_diseases):
    col_idx=idx%num_cols
    cols[col_idx].write(f"☑️ {disease}")


# symptom selection
st.write("Choose 6 unique symptoms for predicting your disease:")
selected_symptoms=st.multiselect(
    "select Symptoms", options=list(df_train.columns[:-1]),max_selections=6
)


if len(selected_symptoms)<6:
    st.warning('Please select exactly 6 symptoms')
else:
    st.write("Take a look at all the Symptoms that you have Selected")
    st.write(",".join(selected_symptoms))

    # set value '1' for selected symptoms and set '0'
#  for those symptoms that are not selected/

all_symptoms=list(df_train.columns[:-1])
symptoms_values=[1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

# predict disease
if st.button("Predict"):
    input_data=pd.DataFrame([symptoms_values],columns=all_symptoms)
    predicted_disease=Decision_Tree_model.predict(input_data)[0]

    # Results
    st.subheader("Predicted Result:")
    st.write(f"You may have: **{predicted_disease}**")

    st.subheader("Selected Symptoms are:")
    st.write(dict(zip(selected_symptoms,[1]*len(selected_symptoms))))
