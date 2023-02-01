import pandas as pd
import numpy as np
import acquire as a
import new_lib as nl

def prepare_wine(df):
    df = df.rename(columns={'fixed_acidity': 'fixed_acidity', 'volatile acidity': 'volatile_acidity', 
                       'citric acid': 'citric_acid', 'residual sugar': 'sugar', 
                       'free sulfur dioxide': 'free_sulfer', 'total sulfur dioxide': 'total_sulfer'})
    return df

def scale_wine(df, s):
    X_train, y_train, X_val, y_val, X_test, y_test = nl.train_vailidate_test_split(df, 'quality')
    train_scaled, val_scaled, test_scaled = nl.scale_splits(X_train, X_val, X_test, scaler = s)
    return X_train, y_train, X_val, y_val, X_test, y_test, train_scaled, val_scaled, test_scaled
