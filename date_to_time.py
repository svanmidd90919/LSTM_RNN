'''
Created on 2019 M10 31

@author: Sheldon Van Middelkoop
'''


def date_to_time(dates):
    """
    Sheldon Van Middelkoop
    
    ------------------------------------------------------------------
    inputs - pandas dateframe with only dates in str format yyyy-mm-dd
    startDate- The first day in the time series in str format
    yyyy-mm-dd
    ------------------------------------------------------------------
    outputs - pandas dataframe with integer number of days from beginning
     of data
     
     NOTE: We will assume the first value is the minimum
    
    """
    for index, row in dates.iterrows():
        # We will take the modulus so we get all time parts for one item in one store
        row['date'] = (index - 1)
        #print(row['date'])
    return dates