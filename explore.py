import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as stats
random_state = 77


def correlations(train):
    plt.figure(figsize=(20, 12))
    sns.heatmap(train.corr(), cmap='Oranges', annot=True)
    plt.title('Train data Columns Correlation', fontsize=14)
    plt.show()


def value_correlations(train):
    '''
    This functino takes in the zillow train sample and uses pandas and seaborn to create a
    ordered list and heatmap of the correlations between the various quantitative feeatures and the target. 
    '''
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(train.corr().abs().logerror).sort_values(by='logerror', ascending=False)
    # rename the correlation column
    corr.columns = ['correlation (abs)']
    # establish figure size
    plt.figure(figsize=(10,10))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True, cmap="mako")
    # establish a plot title
    plt.title('Features\' Correlation with logerror')
    # display the plot
    plt.show()


def LE_dist(df):
    # overall logerror distribution 
    sns.distplot(df.logerror)
    plt.title('Log Error Distribution', fontsize=20)
    plt.xlabel('Log Error')
    plt.show()


def bathroom_viz(df):
    plt.figure(figsize=(14,8))
    with sns.color_palette('Blues'):
        sns.barplot(x='bathroomcnt', y='logerror', data=df)
    plt.xlabel('Bathroom Count')
    plt.ylabel('Log Error')
    plt.title('Does bathroom count impact log error?')
    plt.show()

def bedroom_viz(df):
    plt.figure(figsize=(14,8))
    with sns.color_palette("Blues"):
        sns.barplot(x='bedroomcnt', y='logerror', data=df)
    plt.xlabel('Bedroom Count')
    plt.ylabel('Log Error')
    plt.title('Does bedroom count impact log error?')
    plt.show()

def fast_pearson(x,y):
    alpha = .05

    corr, p = stats.pearsonr(x, y)

    corr, p
    print(f'correlation is {corr}')
    print(f'P-value is {p}')
    if p < alpha:
        print("We reject the null hypothesis")
        print("we have confidence that there is a correlation")
    else:
        print("We fail to reject the null")

def add_clusters(train_scaled, validate_scaled, test_scaled):
    '''
    This function takes in the train, validate, and test samples from the zillow dataset.
    It then performs clustering on various combinations of features in the train sample, 
    Those clusters are then given useful names where appropriate, and added
    as categorical features to the dataset.
    The train, validate, and test df's are returned, in that order.
    '''
    
    # cluster_BedBath

    # identify features
    features = ['bedroomcnt', 'bathroomcnt']
    # create the df to cluster on 
    x = train_scaled[features]
    # create and fit the KMeans object
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    # create cluster labels for each of the samples and add as an additional column
    for sample in [train_scaled, validate_scaled, test_scaled]:
        x = sample[features]
        sample['cluster_BedBath'] = kmeans.predict(x)
        sample['cluster_BedBath'] = sample.cluster_BedBath.map({1:'low', 0:'mid', 2:'high'})

    # repeat the process for each of the desired feature combinations on which to cluster

    # cluster_BedBathSqft

    features = ['bedroomcnt', 'bathroomcnt', 'sq_ft']
    x = train_scaled[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train_scaled, validate_scaled, test_scaled]:
        x = sample[features]
        sample['cluster_BedBathSqft'] = kmeans.predict(x)
        sample['cluster_BedBathSqft'] = sample.cluster_BedBathSqft.map({1:'low', 0:'mid', 2:'high'})

    
    # cluster_BedBathTaxvaluepersqft
    features = ['bedroomcnt', 'bathroomcnt', 'taxval_sqft']
    x = train_scaled[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train_scaled, validate_scaled, test_scaled]:
        x = sample[features]
        sample['cluster_BedBathTaxvaluepersqft'] = kmeans.predict(x)
        sample['cluster_BedBathTaxvaluepersqft'] = sample.cluster_BedBathTaxvaluepersqft.astype(str)
    
        
    # cluster_LatLong
    features = ['latitude', 'longitude']
    x = train_scaled[features]
    kmeans = KMeans(n_clusters=4, random_state=random_state)
    kmeans.fit(x)

    for sample in [train_scaled, validate_scaled, test_scaled]:
        x = sample[features]
        sample['cluster_LatLong'] = kmeans.predict(x)
        sample['cluster_LatLong'] = sample.cluster_LatLong.map({0:'east', 1:'central', 2:'west', 3:'north'})


    return train_scaled, validate_scaled, test_scaled

def viz_cluster_BedBathSqft(train):
    # visualize the clusters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=train, x='bedroomcnt', y='bathroomcnt', hue='cluster_BedBathSqft', palette='bright', ax=ax1)
    sns.scatterplot(data=train, x='bedroomcnt', y='sq_ft', hue='cluster_BedBathSqft', palette='bright', ax=ax2)
    sns.scatterplot(data=train, x='bathroomcnt', y='sq_ft', hue='cluster_BedBathSqft', palette='bright', ax=ax3)
    plt.show()


def test_cluster_BedBathSqft(train):
    # testing whether there is a significant difference in logerror among the clusters

    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_BedBathSqft == 'mid'].logerror,
                             train[train.cluster_BedBathSqft == 'low'].logerror,
                             train[train.cluster_BedBathSqft == 'high'].logerror)
    print()
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0, There is a significant difference in logerror among the BedBathSqft clusters.')
    else: 
        print('Fail to reject H0, There is no significant difference in logerror among the BedBathSqft clusters.')


def viz_cluster_BedBath(train):

    # visualize the clusters
    sns.relplot(data=train, x='bedroomcnt', y='bathroomcnt', hue='cluster_BedBath', palette='bright')
    plt.show()


def test_cluster_BedBath(train):

    # testing whether there is a significant difference in logerror among the clusters
    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_BedBath == 'mid'].logerror, 
                             train[train.cluster_BedBath == 'low'].logerror, 
                             train[train.cluster_BedBath == 'high'].logerror)
    print()
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0, There is a significant difference in logerror among the BedBath clusters.')
    else: 
        print('Fail to reject H0, There is no significant difference in logerror among the BedBath clusters.')       


def viz_clusters_LatLong(train):

    fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True, sharex=True)

    for i, k in enumerate(range(3,9)):

        # creating clusters with KMeans
        x = train[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(x)
        train['cluster'] = kmeans.predict(x)

        # visualize the clusters
        y = int(i / 3)
        x = i % 3
        ax = sns.scatterplot(data=train, x='longitude', y='latitude', hue='cluster',
                             palette='bright', ax=axes[y,x])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)        


def test_cluster_LatLong(train):
    # testing whether there is a significant difference in logerror among the clusters
    alpha = .05
    stat, p = stats.f_oneway(train[train.cluster_LatLong == 'north'].logerror,
                             train[train.cluster_LatLong == 'west'].logerror,
                             train[train.cluster_LatLong == 'central'].logerror,
                             train[train.cluster_LatLong == 'east'].logerror)
    print()
    print()
    print('alpha = ', alpha)
    print()
    print('stat = ', stat)
    print('p = ', p)
    print()
    if p < alpha:
        print('Reject H0, There is a significant difference in logerror among the LatLong clusters.')
    else: 
        print('Fail to reject H0, There is no significant difference in logerror among the LatLong clusters.')