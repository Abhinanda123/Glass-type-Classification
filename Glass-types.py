# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  glass_type= model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])[0]
  glass_list=["building windows float processed","building windows non float processed","vehicle windows float processed","vehicle windows non float processed", "containers", "tableware" ,"headlamp"]
  for i in range(1,8):
    if glass_type== i:
      return glass_list[i-1]

st.sidebar.title("Use-based Glass Classification")
st.title("Glass Classification")

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Dataset for Glass Classification")
  st.dataframe(glass_df)

st.sidebar.subheader("Plot the Data")
list_name= st.sidebar.multiselect("Plot options",("Co-relation Heatmap", "Line Chart", "Area Chart", "Count plot", "Pie chart", "Box plot"))

st.set_option('deprecation.showPyplotGlobalUse', False)

if "Line Chart" in list_name:
  st.subheader("Line Chart")
  st.line_chart(glass_df)
if "Area Chart" in list_name:
  st.subheader("Area Chart")
  st.area_chart(glass_df)
if "Co-relation Heatmap" in list_name:
  st.subheader("Co-relation Heatmap")
  plt.figure(figsize=(10,10))
  sns.heatmap(glass_df.corr(), annot=True)
  st.pyplot()
if "Count plot" in list_name:
  st.subheader("Count plot")
  plt.figure(figsize=(10,10))
  sns.countplot(glass_df["GlassType"])
  st.pyplot()
if "Pie chart" in list_name:
  st.subheader("Pie chart")
  plt.figure(figsize=(10,10))
  plt.pie(glass_df["GlassType"].value_counts())
  st.pyplot()
if "Box plot" in list_name:
  st.subheader("Box plot")
  column= st.sidebar.selectbox("Select the column", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,10))
  sns.boxplot(glass_df[column])
  st.pyplot()

st.sidebar.subheader("Select Your Values")
RI= st.sidebar.slider("Refractive Index", float(glass_df["RI"].min()),float(glass_df["RI"].max()))
Na= st.sidebar.slider("Na", float(glass_df["Na"].min()),float(glass_df["Na"].max()))
Mg= st.sidebar.slider("Mg", float(glass_df["Mg"].min()),float(glass_df["Mg"].max()))
Al= st.sidebar.slider("Al", float(glass_df["Al"].min()),float(glass_df["Al"].max()))
Si= st.sidebar.slider("SI", float(glass_df["Si"].min()),float(glass_df["Si"].max()))
K= st.sidebar.slider("K", float(glass_df["K"].min()),float(glass_df["K"].max()))
Ca= st.sidebar.slider("Ca", float(glass_df["Ca"].min()),float(glass_df["Ca"].max()))
Ba= st.sidebar.slider("Ba", float(glass_df["Ba"].min()),float(glass_df["Ba"].max()))
Fe= st.sidebar.slider("Fe", float(glass_df["Fe"].min()),float(glass_df["Fe"].max()))

st.sidebar.subheader("Choose Classifier")
classifier= st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier',"Logistic Regression"))

from sklearn.metrics import plot_confusion_matrix
if classifier== "Support Vector Machine":
  st.sidebar.subheader("Model Hyperparameters")
  c= st.sidebar.number_input("C (Error Rates)", 1,100,1)
  kernel= st.sidebar.radio("Kernel", ("linear","rbf","poly"))
  gamma= st.sidebar.number_input("Gamma", 1,100,1)
  if st.sidebar.button("Classify"):
    st.subheader("Support Vector Machine")
    svc= SVC(C= c,kernel= kernel, gamma= gamma)
    svc.fit(X_train, y_train)
    pred= prediction(svc, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
    score= svc.score(X_train,y_train)
    st.write("The Glass Type predicted is: ", pred)
    st.write("The accuracy of the model is: ", score)
    plot_confusion_matrix(svc, X_test, y_test)
    st.pyplot()

if classifier== "Random Forest Classifier":
  st.sidebar.subheader("Model Hyperparameters")
  n_estimators= st.sidebar.number_input("Number of trees in the forest", 20,10000, step= 50)
  max_depth= st.sidebar.number_input("Maximum depth of the tree", 1,100,1)
  if st.sidebar.button("Classify"):
    st.subheader("Random Forest Classifier")
    rfc= RandomForestClassifier(n_estimators=n_estimators, n_jobs= -1, max_depth=max_depth)
    rfc.fit(X_train, y_train)
    pred= prediction(rfc, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
    score= rfc.score(X_train,y_train)
    st.write("The Glass Type predicted is: ", pred)
    st.write("The accuracy of the model is: ", score)
    plot_confusion_matrix(rfc, X_test, y_test)
    st.pyplot()

if classifier== "Logistic Regression":
  st.sidebar.subheader("Model Hyperparameters")
  max_iter= st.sidebar.number_input("Number of iterations", 100,1000,step= 10)
  c= st.sidebar.number_input("C (Error rates)", 1,100,step= 1)
  if st.sidebar.button("Classify"):
    st.subheader("Logistic Regression")
    lgc= LogisticRegression(max_iter= max_iter, C=c)
    lgc.fit(X_train, y_train)
    pred= prediction(lgc, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
    score= lgc.score(X_train,y_train)
    st.write("The Glass Type predicted is: ", pred)
    st.write("The accuracy of the model is: ", score)
    plot_confusion_matrix(lgc, X_test, y_test)
    st.pyplot()