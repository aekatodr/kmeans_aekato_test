import streamlit as st
import pandas as pd
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model from the file
kmeans_model = joblib.load('bmx_kmeans.joblib')

st.title('K-Means Clustering')

# Upload the dataset and save as csv
uploaded_file = st.file_uploader("Choose a CSV file", type ="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # select the columns to be used for clustering
    data = data[['bmxleg','bmxwaist']].dropna()

    # Check if there 

    # Predict the cluster for each data point
    clusters = kmeans_model.predict(data)

    # add cluster labels to the datafame
    data['cluster'] = clusters

    st.write(data)

    # Plot the clusters
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Scatterplot of bmxleg and bmxwaist")
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='bmxleg', y='bmxwaist', hue='cluster')
    st.pyplot()
else:
    st.write("Not enough data points to cluster")