English description below


Algoritmos de Clusterização

Este repositório contém implementações de técnicas de clusterização utilizando algoritmos populares em aprendizado de máquina não supervisionado. 

Há três arquivos principais, cada um com foco em diferentes métodos e abordagens de agrupamento: KMeans, KMeans Semi-Supervisionado, e DBSCAN.

Arquivos no Repositório

1. KMeans
   
O arquivo aborda o uso do algoritmo de clusterização KMeans aplicado em diferentes contextos e com diversas técnicas de análise. 

As principais etapas e funcionalidades incluem:

Geração de Dados Sintéticos: 

Utilizei make_blobs para criar conjuntos de dados com múltiplos clusters.

Clusterização com KMeans: 

Agrupamento de dados sintéticos em clusters definidos.

Predição de Novos Dados: 

Determinei os clusters para novos pontos de dados.

Tesselação de Voronoi: 

Visualizei os limites entre os clusters gerados.

Segmentação de Imagem: 

Apliquei KMeans para segmentação de uma imagem em diferentes números de clusters.

Análise de Silhouette: 

Avaliei a qualidade dos clusters gerados.

Curva de Inércia e Avaliação de Clusters: 

Explorei diferentes números de clusters para otimizar o agrupamento.


Kmeans-Semi-Supervised-Learning:

Este projeto explora técnicas de aprendizado semi-supervisionado, 
combinando o uso de algoritmos de clusterização e aprendizado supervisionado para melhorar a acurácia de modelos em cenários com dados rotulados limitados. 
A abordagem principal envolve o uso do algoritmo K-Means para selecionar instâncias representativas de clusters, que são então rotuladas e usadas para treinar modelos supervisionados.

Principais Etapas:

Pré-processamento e Preparação dos Dados:

Usei o dataset load_digits da biblioteca Scikit-learn.

Separação dos dados em conjuntos de treino e teste.

Clusterização com K-Means:

Aplicação do K-Means para dividir os dados em 30 clusters.

Identificação de instâncias representativas de cada cluster.

Visualização dos clusters em um gráfico.

Treinamento Inicial Supervisionado:

Treinamento de um modelo de Regressão Logística usando apenas as instâncias representativas.

Propagação de Rótulos:

Propagação dos rótulos das instâncias representativas para todos os dados de treinamento, com base nos clusters do K-Means.

Modelos Semi-Supervisionados:

Comparação de diferentes abordagens semi-supervisionadas, incluindo:

Label Spreading

Label Propagation

SelfTrainingClassifier com um classificador de floresta aleatória.

Avaliação das performances usando métricas de acurácia.

Active Learning:

Implementação de uma abordagem de aprendizado ativo, onde o modelo identifica e solicita rótulos para instâncias não rotuladas mais incertas.

Aumento progressivo da acurácia ao longo de múltiplas rodadas de aprendizado.




DBSCAN

Neste código, usei o algoritmo DBSCAN para explorar a clusterização de dados sintéticos e reais. 
Minha ideia foi testar como ele se comporta em diferentes cenários e integrar algumas técnicas adicionais para complementar os resultados. 

Aqui está um resumo do que fiz:

Dados Sintéticos (make_moons)

Comecei com um dataset em formato de "meia-lua" com ruído.

Ajustei os parâmetros eps e min_samples para ver como o DBSCAN identifica clusters e ruídos.

Visualizei os clusters formados e os pontos considerados ruído, além de explorar como esses parâmetros influenciam os resultados.

Classificação de Novos Pontos

Treinei um KNeighborsClassifier usando os "core points" gerados pelo DBSCAN.

Testei várias métricas de distância (Euclidiana, Manhattan, Minkowski, Chebyshev).

Depois, classifiquei novos pontos e calculei as probabilidades associadas a cada classe, comparando os resultados.

HDBSCAN e Comparações

Experimentei o HDBSCAN para verificar como ele se sai com parâmetros automáticos.

Comparei a clusterização com outros métodos, como KMeans e Agglomerative Clustering.

Usei o Silhouette Score para medir a qualidade de cada abordagem.

Dados Reais (Mall_Customers.csv)

Apliquei o DBSCAN em um dataset real com características como "Idade", "Renda Anual" e "Pontuação de Gastos".

Escalei os dados para melhorar o desempenho do algoritmo.

Visualizei os clusters em um gráfico 3D e analisei como o DBSCAN lida com distribuições reais e multidimensionais.

Explorações Adicionais

Testei outros algoritmos, como KMeans, BIRCH, MeanShift e Spectral Clustering, para entender as diferenças práticas.

Variações de parâmetros como eps no DBSCAN, k no KMeans e linkage no Agglomerative Clustering foram avaliadas para ver como afetam os clusters formados.

Visualizações

Criei gráficos em 2D e 3D para visualizar os resultados de cada abordagem. 
Isso ajudou a identificar padrões nos dados e a verificar como cada algoritmo agrupa os pontos.








Clustering Algorithms

This repository contains implementations of clustering techniques using popular algorithms in unsupervised machine learning.

There are three main files, each focusing on different methods and clustering approaches: 

KMeans, Semi-Supervised KMeans, and DBSCAN.

Repository Files

1. KMeans
   
This file explores the use of the KMeans clustering algorithm applied in various contexts and with diverse analytical techniques.

Key steps and functionalities include:

Synthetic Data Generation:

I used make_blobs to create datasets with multiple clusters.

Clustering with KMeans:

I grouped synthetic data into defined clusters.

Prediction on New Data:

I determined the cluster labels for new data points.

Voronoi Tessellation:

I visualized boundaries between the generated clusters.

Image Segmentation:

Applied KMeans to segment an image into different numbers of clusters.

Silhouette Analysis:

Evaluated the quality of the generated clusters.

Inertia Curve and Cluster Evaluation:

Explored different cluster numbers to optimize clustering.

2. KMeans-Semi-Supervised-Learning
   
This project explores semi-supervised learning techniques by combining clustering algorithms with supervised learning to improve model accuracy in scenarios with limited labeled data. 
The primary approach involves using the KMeans algorithm to select representative instances of clusters, which are then labeled and used to train supervised models.

Key Steps:

Data Preprocessing and Preparation:

Used the load_digits dataset from Scikit-learn.

Split the data into training and testing sets.

Clustering with KMeans:

Applied KMeans to divide the data into 30 clusters.

I identified representative instances for each cluster.

I visualized clusters on a plot.

Initial Supervised Training:

I trained a Logistic Regression model using only the representative instances.

Label Propagation:

Propagated labels from representative instances to all training data based on KMeans clusters.

Semi-Supervised Models:

Compared different semi-supervised approaches, including:

Label Spreading

Label Propagation

SelfTrainingClassifier with a Random Forest classifier.

Evaluated performance using accuracy metrics.

Active Learning:

Implemented an active learning approach where the model identifies and requests labels for the most uncertain unlabeled instances.

Achieved progressive accuracy improvements over multiple learning rounds.

3. DBSCAN
   
This file utilizes the DBSCAN algorithm to explore clustering of both synthetic and real-world data.
The goal was to test its performance in various scenarios and integrate additional techniques to complement the results.

Summary:

Synthetic Data (make_moons):

Started with a "half-moon" shaped dataset with noise.

Tuned eps and min_samples to analyze how DBSCAN identifies clusters and noise.

Visualized the formed clusters and noise points, and explored parameter impacts on results.

Classifying New Points:

Trained a KNeighborsClassifier using the "core points" generated by DBSCAN.

Tested various distance metrics (Euclidean, Manhattan, Minkowski, Chebyshev).

Classified new points and computed class probabilities, comparing results.

HDBSCAN and Comparisons:

Experimented with HDBSCAN to evaluate its performance with automatic parameters.

Compared clustering results with other methods, such as KMeans and Agglomerative Clustering.

Used Silhouette Score to measure the quality of each approach.

Real Data (Mall_Customers.csv):

Applied DBSCAN on a real dataset with features like "Age," "Annual Income," and "Spending Score."

Scaled the data to improve algorithm performance.

Visualized clusters in a 3D plot and analyzed DBSCAN's handling of real and multidimensional distributions.

Additional Explorations:

Tested other algorithms like KMeans, BIRCH, MeanShift, and Spectral Clustering to understand practical differences.

Evaluated parameter variations such as eps in DBSCAN, k in KMeans, and linkage in Agglomerative Clustering to observe effects on clustering results.

Visualizations:

Created 2D and 3D plots to visualize the results of each approach.
It helped identify data patterns and verify how each algorithm groups points.
