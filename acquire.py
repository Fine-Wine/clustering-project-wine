#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### ACQUIRE ###


def get_wine_data():
    '''
    This function reads the wine data from 
    https://data.world/food/wine-quality site into a df.
    '''
    import pandas as pd

    # reads in the red wine data set and assigns the df to a variable
    red_wine_df = pd.read_csv('https://query.data.world/s/bqmesefq3qbzhzjyyjd52u7lfntuzv')
    
    # creates a column named red_or_white and assigns red wine to 1 for yes
    red_wine_df.insert(0, 'red_or_white', 1)

    # reads in the white wine data set and assigns the df to a variable
    white_wine_df = pd.read_csv('https://query.data.world/s/hqayd6nqceg6wkq3ziib6ostwmw4pz')

    # creates a column named red_or_white and assigns white wine to 0 for no
    white_wine_df.insert(0, 'red_or_white', 0)

    # appends the red and white wine dataframes to one dataframe that includes the new
    # column red_or_white
    wine_df = red_wine_df.append(white_wine_df, ignore_index=True)
    
    # returns the appended wine dataframe
    return wine_df



def acquire_wine():
    '''
    This function reads in wine data from the 
    https://data.world/food/wine-quality site, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    import os
    
    if os.path.isfile('wine.csv'):
        
        # If csv file exists, read in data from csv file.
        wine_df = pd.read_csv('wine.csv', index_col=0)
        
    else:

        #creates new csv if one does not already exist
        wine_df = get_wine_data()
        wine_df.to_csv('wine.csv')

    return wine_df

