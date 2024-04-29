# import os
# import nltk
# import networkx as nx
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt

# # Download NLTK resources if necessary
# nltk.download('punkt')
# nltk.download('stopwords')

# # Define global variables
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# # Function to preprocess text
# def preprocess_text(text):
#     tokens = word_tokenize(text.lower())
#     tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
#     return tokens

# # Function to construct graph from text
# def construct_graph(text):
#     tokens = preprocess_text(text)
#     graph = nx.DiGraph()
#     for i in range(len(tokens) - 1):
#         word1, word2 = tokens[i], tokens[i+1]
#         if not graph.has_edge(word1, word2):
#             graph.add_edge(word1, word2, weight=1)
#         else:
#             graph[word1][word2]['weight'] += 1
#     return graph

# folder_path1 = r'C:\Mahwish\GT\fashion&beauty\preprocessed_data'
# folder_path2=  r'C:\Mahwish\GT\Health&fitness\preprocessed_data'
# # folder_path3=  r'E:\6th semester\GT\Business&Finance\preprocess_data'
# # Load data for each topic
# def load_data(folder_path1):
#     data = []
#     for i in range(1, 13):  # Loop from 1 to 12
#         filename = f"cleaned_data_{i}.txt"  # Add file extension '.txt'
#         with open(os.path.join(folder_path1, filename), 'r', encoding='utf-8') as file:
#             text = file.read()
#             data.append(text)
#     return data
# # load data for topic 2
# def load_data2(folder_path2):
#     data = []
#     for i in range(1, 13):  # Loop from 1 to 12
#         filename = f"cleaned_data_{i}.txt"  # Add file extension '.txt'
#         with open(os.path.join(folder_path2, filename), 'r', encoding='utf-8') as file:
#             text = file.read()
#             data.append(text)
#     return data
# #load data for topic 3
# # def load_data3(folder_path3):
# #     data = []
# #     for i in range(1, 13):  # Loop from 1 to 12
# #         filename = f"cleaned_data_{i}.txt"  # Add file extension '.txt'
# #         with open(os.path.join(folder_path3, filename), 'r', encoding='utf-8') as file:
# #             text = file.read()
# #             data.append(text)
# #     return data

# # Function to preprocess and construct graphs for training data
# def preprocess_and_construct_graphs(data):
#     graphs = []
#     for text in data:
#         graph = construct_graph(text)
#         graphs.append(graph)
#     return graphs

# # Function to extract features from graphs
# def extract_features(graphs):
#     features = []
#     for graph in graphs:
#         # Extract features from the graph (example feature: number of nodes)
#         features.append(len(graph.nodes))
#     return features

# if __name__ == "__main__":
#     # Data Collection and Preparation
#     topic1_data = load_data(folder_path1)
#     topic2_data = load_data2(folder_path2)
#     # topic3_data = load_data3(folder_path3)

#     # Construct graphs for training data
#     topic1_graphs = preprocess_and_construct_graphs(topic1_data)
#     topic2_graphs = preprocess_and_construct_graphs(topic2_data)
#     # topic3_graphs = preprocess_and_construct_graphs(topic3_data)

#     # Extract features from graphs
#     topic1_features = extract_features(topic1_graphs)
#     topic2_features = extract_features(topic2_graphs)
#     # topic3_features = extract_features(topic3_graphs)

#     # Prepare training data and labels
#     # X_train = topic1_features[:12] + topic2_features[:12] + topic3_features[:12]
#     # y_train = [1] * 12 + [2] * 12 + [3] * 12
    
#     #for two topic
#     X_train = topic1_features[:12] + topic2_features[:12] 
#     y_train = [1] * 12 + [2] * 12 
#     # Prepare test data and labels
#     # X_test = topic1_features[12:] + topic2_features[12:] + topic3_features[12:]
#     # y_test = [1] * 3 + [2] * 3 + [3] * 3
#     X_test = topic1_features[12:] + topic2_features[12:] 
#     y_test = [1] * 3 + [2] * 3 


        

#     # # Reshape your input data
#     # X_train_reshaped = np.array(X_train).reshape(-1, 1)
#     # y_train_reshaped = np.array(y_train).reshape(-1, 1)
#     # X_test_reshaped = np.array(X_test).reshape(-1, 1)
#     # y_test_reshaped = np.array(y_test).reshape(-1, 1)

#     # # Train KNN classifier
#     # knn_classifier = KNeighborsClassifier(n_neighbors=3)
#     # knn_classifier.fit(X_train_reshaped, y_train_reshaped)

#     #     # Flatten the y array using ravel()
#     # y_train_flat = np.ravel(y_train)
#     # y_test_flat = np.ravel(y_test)

#     # # Train KNN classifier
#     # knn_classifier = KNeighborsClassifier(n_neighbors=3)
#     # knn_classifier.fit(X_train, y_train_flat)

#     # Train KNN classifier
#     knn_classifier = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
#     knn_classifier.fit(X_train, y_train)

#     # Predict
#     y_pred = knn_classifier.predict(X_test)

#     # Evaluate
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)

#     # Classification report
#     print(classification_report(y_test, y_pred))

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

import os
import nltk
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Download NLTK resources if necessary
nltk.download('punkt')
nltk.download('stopwords')

# Define global variables
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Function to construct graph from text
def construct_graph(text):
    tokens = preprocess_text(text)
    graph = nx.Graph()
    for i in range(len(tokens) - 1):
        word1, word2 = tokens[i], tokens[i+1]
        if not graph.has_edge(word1, word2):
            graph.add_edge(word1, word2, weight=1)
        else:
            graph[word1][word2]['weight'] += 1
    return graph

# Function to apply Minimum Connected Subgraph (MCS) algorithm
def apply_mcs(graph):
    # Convert the directed graph to an undirected graph
    undirected_graph = graph.to_undirected()
    # Apply Minimum Spanning Tree algorithm
    mcs_graph = nx.minimum_spanning_tree(undirected_graph)
    return mcs_graph

folder_path1 = r'C:\Mahwish\GT\fashion&beauty\preprocessed_data'
folder_path2 = r'C:\Mahwish\GT\Health&fitness\preprocessed_data'

# Load data for each topic
def load_data(folder_path):
    data = []
    for i in range(1, 13):  # Loop from 1 to 12
        filename = f"cleaned_data_{i}.txt"  # Add file extension '.txt'
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            data.append(text)
    return data

# Function to preprocess and construct graphs for training data
def preprocess_and_construct_graphs(data):
    graphs = []
    for text in data:
        graph = construct_graph(text)
        mcs_graph = apply_mcs(graph)  # Apply MCS algorithm
        graphs.append(mcs_graph)
    return graphs

# Function to extract features from graphs
def extract_features(graphs):
    features = []
    for graph in graphs:
        # Extract features from the graph (example feature: number of nodes)
        features.append(len(graph.nodes))
    return features

if __name__ == "__main__":
    # Data Collection and Preparation
    topic1_data = load_data(folder_path1)
    topic2_data = load_data(folder_path2)

    # Construct graphs for training data
    topic1_graphs = preprocess_and_construct_graphs(topic1_data)
    topic2_graphs = preprocess_and_construct_graphs(topic2_data)

    # Extract features from graphs
    topic1_features = extract_features(topic1_graphs)
    topic2_features = extract_features(topic2_graphs)

    # Prepare training data and labels
    X_train = topic1_features[:12] + topic2_features[:12]
    y_train = [1] * 12 + [2] * 12

    # Prepare test data and labels
    X_test = topic1_features[12:] + topic2_features[12:]
    y_test = [1] * 3 + [2] * 3

    # Reshape the input features (X_train and X_test)
    X_train_reshaped = np.array(X_train).reshape(-1, 1)
    X_test_reshaped = np.array(X_test).reshape(-1, 1)

    # Check the shapes of training and test data
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", len(y_train))
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", len(y_test))

    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train_reshaped, y_train)

    # Predict
    y_pred = knn_classifier.predict(X_test_reshaped)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Classification report
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
