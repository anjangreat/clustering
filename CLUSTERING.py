# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Loading dataset
data = pd.read_csv('data/winequality-red.csv', delimiter=';')

# Handling missing values
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data
def clustering_pipeline(X, n_clusters, technique='kmeans', preprocessing=None):
    # Preprocessing
    if preprocessing == 'normalize':
        X = StandardScaler().fit_transform(X)
    elif preprocessing == 'transform':
        X = MinMaxScaler().fit_transform(X)
    elif preprocessing == 'pca':
        X = PCA(n_components=2).fit_transform(X)
    elif preprocessing == 't+n':
        X = StandardScaler().fit_transform(X)
        X = PCA(n_components=2).fit_transform(X)
    
    # Applying clustering
    if technique == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif technique == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif technique == 'meanshift':
        model = MeanShift()
    
    labels = model.fit_predict(X)

    # Evaluating clustering performance
    silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else 'NA'
    calinski = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 'NA'
    davies = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else 'NA'
    
    return silhouette, calinski, davies
# Parameters for experiments
cluster_counts = [3, 4, 5]
preprocessing_methods = [None, 'normalize', 'transform', 'pca', 't+n']
techniques = ['kmeans', 'hierarchical', 'meanshift']
results = []

# Looping through each combination
for technique in techniques:
    for n_clusters in cluster_counts:
        for method in preprocessing_methods:
            silhouette, calinski, davies = clustering_pipeline(X, n_clusters, technique, method)
            results.append({
                'Technique': technique,
                'Preprocessing': method,
                'Clusters': n_clusters,
                'Silhouette': silhouette,
                'Calinski-Harabasz': calinski,
                'Davies-Bouldin': davies
            })
            results_df = pd.DataFrame(results)

# Print results in table format
print(results_df)
techniques = results_df['Technique'].unique()

# Converting result to styled table for better visualization
styled_tables = []
for technique in techniques:
    subset_df = results_df[results_df['Technique'] == technique]
    pivoted_df = subset_df.pivot(index='Clusters', columns='Preprocessing', values=['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'])
    
    styled_table = pivoted_df.style.set_caption(f"Performance of {technique.capitalize()} Clustering") \
                   .background_gradient(cmap="Blues", subset=['Silhouette']) \
                   .background_gradient(cmap="Blues", subset=['Calinski-Harabasz']) \
                   .background_gradient(cmap="Blues", subset=['Davies-Bouldin'])
    
    styled_tables.append(styled_table)

for table in styled_tables:
    display(table)

sns.catplot(data=results_df, x='Clusters', y='Silhouette', hue='Preprocessing', col='Technique', kind='bar')
plt.show()
    