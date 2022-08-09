import os
import pandas as pd
import env
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
random_state = 77

################

### GLOBALS ###

query = '''
    SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL
      AND prop.longitude IS NOT NULL
      AND transactiondate <= '2017-12-31'
      AND propertylandusedesc = "Single Family Residential"
        '''

###############

### FUNCTIONS ###

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print()
    print('--- Info')
    df.info()
    print()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

def acquire():
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv', index=False)
    return df


def tidy(df):
    # drop redundant id code columns
    id_cols = [col for col in df.columns if 'typeid' in col or col in ['id', 'parcelid']]
    df = df.drop(columns=id_cols)
    # filter for single family properties
    df = df[df.propertylandusedesc == 'Single Family Residential']
    # drop specified columns
    cols_to_drop = ['calculatedbathnbr',
                    'finishedfloor1squarefeet',
                    'finishedsquarefeet12', 
                    'regionidcity',
                    'landtaxvaluedollarcnt',
                    'taxamount',
                    'rawcensustractandblock',
                    'roomcnt',
                    'regionidcounty',
                    'propertycountylandusecode',
                    'regionidzip',
                    'transactiondate',
                    'censustractandblock',
                    'threequarterbathnbr',
                    'assessmentyear']
    df = df.drop(columns=cols_to_drop)
    # fill null values with 0 in specified columns
    cols_to_fill_zero = ['fireplacecnt',
                         'garagecarcnt',
                         'garagetotalsqft',
                         'hashottuborspa',
                         'poolcnt',
                         'taxdelinquencyflag']
    for col in cols_to_fill_zero:
        df[col] = np.where(df[col].isna(), 0, df[col]) 
    # drop columns with more than 5% null values
    for col in df.columns:
        if df[col].isnull().mean() > .05:
            df = df.drop(columns=col)
    # drop duplicate rows and remaining nulls
    df = df.drop_duplicates()
    df = df.dropna()
    
    return df



def optimize(df):
    #Create a dictionary mapping the fips values to the counties
    county_dict = {6037.0: 'LA County', 6059.0: 'Orange County',
                   6111.0: 'Ventura County'}
    #Replace the fips numbers with county names
    df['fips'].replace(county_dict, inplace=True)
    df.rename(columns={'fips': 'county'}, inplace=True)
    # change the 'Y' in taxdelinquencyflag to 1
    df['taxdelinquencyflag'] = np.where(df.taxdelinquencyflag == 'Y', 1, df.taxdelinquencyflag)
    # change boolean column to int
    df['hot_tub_or_spa'] = df.hashottuborspa.apply(lambda x: str(int(x)))
    # changing year from float to int
    df['yearbuilt'] = df.yearbuilt.apply(lambda x: int(x))
    # moving the latitude and longitude decimal place
    df['latitude'] = df.latitude / 1_000_000
    df['longitude'] = df.longitude / 1_000_000
    # adding a feature: age 
    df['age'] = 2017 - df.yearbuilt
    df = df.drop(columns='yearbuilt')
    # add a feature: taxvalue_per_sqft
    df['taxval_sqft'] = df.taxvaluedollarcnt / df.calculatedfinishedsquarefeet
    # rename sqft column
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sq_ft'})
    
    
    return df

def train_validate_test_split(df):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    # split the dataframe into train and test
    train, test = train_test_split(df, test_size=.2, random_state=77)
    # further split the train dataframe into train and validate
    train, validate = train_test_split(train, test_size=.3, random_state=77)
    # print the sample size of each resulting dataframe
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test




def MM_scale_zillow(train, validate, test):
    '''
    This takes in the train, validate, and test dataframes, as well as the target label. 
    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the train, validate, and test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 
    train, validate, and test dataframes are returned, in that order. 
    '''
    target = 'logerror'
    
    # identify quantitative features to scale
    quant_features = [col for col in train.columns if (train[col].dtype != 'object') 
                                                    & (col != target)]
    
    # establish empty dataframes for storing scaled dataset
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # create and fit the scaler
    scaler = MinMaxScaler().fit(train[quant_features])
    
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
   

    return train_scaled, validate_scaled, test_scaled

def dummies(train, validate, test):
    # Get dummies for non-binary categorical variables
    dummy_train = pd.get_dummies(train[['cluster_BedBath',\
                                'cluster_BedBathSqft',\
                                'cluster_BedBathTaxvaluepersqft',\
                                'cluster_LatLong']], dummy_na=False, \
                              drop_first=True)

    dummy_validate = pd.get_dummies(validate[['cluster_BedBath',\
                                'cluster_BedBathSqft',\
                                'cluster_BedBathTaxvaluepersqft',\
                                'cluster_LatLong']], dummy_na=False, \
                              drop_first=True)

    dummy_test = pd.get_dummies(test[['cluster_BedBath',\
                                'cluster_BedBathSqft',\
                                'cluster_BedBathTaxvaluepersqft',\
                                'cluster_LatLong']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)

    return train, validate, test

def prepare(df):

    df = tidy(df)

    df = optimize(df)

    return df

