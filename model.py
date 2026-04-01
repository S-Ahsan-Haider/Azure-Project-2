import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle as pkl


df = pd.read_csv(r'D:\Code Related Files\loan_dataset_azure_train.csv')

df = df.drop('Loan_ID', axis=1)
x = df.drop('Loan_Status', axis=1)
y = df['Loan_Status'].map({'Y':1, 'N':0})

num = x.select_dtypes(include=['int64', 'float64']).columns
cat = x.select_dtypes(include=['object']).columns

num_pi = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='median')),
           ('scaler', StandardScaler())]
)

cat_pi = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')),
           ('onehot', OneHotEncoder(handle_unknown='ignore'))]
)

# Combine 
prepro = ColumnTransformer(
    transformers = [
        ('num', num_pi, num),
        ('cat', cat_pi, cat)
    ])

# Pipeline (full)
pipe = Pipeline(steps=[
    ('preprocessor', prepro),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
pipe.fit(xtrain, ytrain)

with open('loan_model.pkl', 'wb') as f:
    pkl.dump(pipe, f)

print("Preprocessing done, model trained and saved!")

#