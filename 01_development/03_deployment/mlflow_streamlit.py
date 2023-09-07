# Import Libraries

# +++++++++++++++++++++++++++++
#  general Purpose
# +++++++++++++++++++++++++++++
import streamlit as st  
import pandas as pd
import numpy as np
import subprocess
import os
import webbrowser

# +++++++++++++++++++++++++++++
#  ML Model
# +++++++++++++++++++++++++++++

#data preparation
from sklearn.model_selection import train_test_split

#Model
from sklearn.linear_model import Lasso

#MLOPS
import mlflow


# Configure Page
st.set_page_config(
    page_title="Boston Housing",
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded") 

# load feature extracted data
df = pd.read_csv("./Data/Boston.csv")

# HELPER FUNCTIONS
# Add the data preprocessor that you need

def preprocess():
    
    return 

# Train the model
def train_model(exp_name, df, a, c):     
    
    # Split the data into features (X) and labels (y)
    X=df.drop(['medv'], axis=1)
    y=df['medv']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create or Select Experiment 
    experiment = mlflow.set_experiment(exp_name)    
    with mlflow.start_run(experiment_id=experiment.experiment_id):          
        # Model
        lasso = Lasso()
        lasso001 = Lasso(alpha=a, max_iter=c).fit(X_train, y_train)
        
        # Make predictions on the training & test set
        # y_train_pred = rf_classifier.predict(x_train_vectorized)
        # y_test_pred = rf_classifier.predict(x_test_vectorized)


        # Evaluate the model
        train_acc = lasso001.score(X_train, y_train)
        test_acc = lasso001.score(X_test, y_test)
                
        # Log Parameters & Metrics
        mlflow.log_params({"alpha":a, "max_inter": c})        
        mlflow.log_metrics({"Training Accuracy": train_acc, "Test Accuracy": test_acc})
        # Log Model & Vectorizer
        mlflow.sklearn.log_model(lasso001, "model")
        # mlflow.sklearn.log_model(vectorizer, "vectorizer") 
    return train_acc, test_acc

# Function for opening MLFlow UI directly from Streamlit
def open_mlflow_ui():
    # Start the MLflow tracking server as a subprocess
    cmd = "mlflow ui --port 5000"
    subprocess.Popen(cmd, shell=True)
def open_browser(url):
    webbrowser.open_new_tab(url)
    
# STREAMLIT UI   
# Sidebar for hyperparameter tuning
st.sidebar.title("Tune Hyper Params ⚙️")
a = st.sidebar.slider('alpha',min_value=0.001, max_value=1.0, step=0.0005, value=0.01)
c = st.sidebar.slider('Max_inter', min_value=1000, max_value=100000, step=1000, value=10000)
# c = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'], index=1)

# Launch Mlflow from Streamlit
st.sidebar.title("Mlflow Tracking 🔎")    
if st.sidebar.button("Launch 🚀"):
    open_mlflow_ui()
    st.sidebar.success("MLflow Server is Live! http://localhost:5000")
    open_browser("http://localhost:5000")

# Main Page Content
st.title("House Boston 🤖")
exp_type = st.radio("Select Experiment Type", ['New Experiment', 'Existing Experiment'], horizontal=True)
if exp_type == 'New Experiment':
    exp_name = st.text_input("Enter the name for New Experiment")
else:
    try:
        if os.path.exists('./mlruns'):
            exps = [i.name for i in mlflow.search_experiments()]
            exp_name = st.selectbox("Select Experiment", exps)
        else:
            st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")            
    except:
        st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")

# Training the model starts from here    
if st.button("Train ⚙️"):
    with st.spinner('Feeding the data--->🧠'):
        tr_a, ts_a = train_model(exp_name, df, a, c)
    st.success('Trained!') 
    st.write(f"Training Accuracy Achieved: {tr_a:.3f}")      