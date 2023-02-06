def eval_result(df, col):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    
    t, p = stats.ttest_1samp(df[col][df.quality>= 6], df[col].mean())
    
    null_hypothesis = "alchol content does not effect the quality of the wine."
    alternative_hypothesis = "alcohol content effects the quality of the wine."
    
    if p/2 <= .05 and t != 0:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
def eval_result2(df, col):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    
    t, p = stats.ttest_1samp(df[col][df.quality>= 6], df[col].mean())
    
    null_hypothesis = "density levels do not effect the quality of wine.  "
    alternative_hypothesis = "density levels effect the quality of wine."
    
    if p/2 <= .05 and t != 0:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
def eval_result3(df, col):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    
    t, p = stats.ttest_1samp(df[col][df.quality>= 6], df[col].mean())
    
    null_hypothesis = "sugar has no effect on determining the quality of the wine."
    alternative_hypothesis = "the different levels of sugar effect the quality of the wine."
    
    if p/2 <= .05 and t != 0:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null") 
        
        
def eval_result4(df, col):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    
    t, p = stats.ttest_1samp(df[col][df.quality>= 6], df[col].mean())
    
    null_hypothesis = "the acidity level of wine does not effect the quality of the wine.  "
    alternative_hypothesis = "the acidity level of wine effects the quality of the wine."
    
    if p/2 <= .05 and t != 0:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null") 
        
        
        
        
def wine_df_scaled(df):
    '''
    This function takes in a dataframe and creates a new scaled dataframe that also includes the 
    clustered features and reutrns this new dataframe.
    '''
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    
    df = df[['volatile_acidity', 'citric_acid', 'sugar', 'density', 'alcohol', 'quality']]
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=['volatile_acidity', 'citric_acid', 'sugar', 'density', 'alcohol', 'quality'])
    
    X = df[['volatile_acidity', 'alcohol']]
    kmeans = KMeans(n_clusters=3, random_state = 77)
    kmeans.fit(X)
    kmeans.predict(X)
    # creating a kmeans and fitting it
    df['v_a'] = kmeans.predict(X)
    
    Y = df[['density', 'alcohol']]
    kmeans = KMeans(n_clusters=3, random_state = 42)
    kmeans.fit(Y)
    kmeans.predict(Y)
    # creating a kmeans and fitting it
    df['d_a'] = kmeans.predict(Y)
    
    Z = df[['sugar', 'density']]
    kmeans = KMeans(n_clusters=3, random_state = 41)
    kmeans.fit(Z)
    kmeans.predict(Z)
    # creating a kmeans and fitting it
    df['s_d'] = kmeans.predict(Z)
    
    A = df[['volatile_acidity', 'citric_acid']]
    kmeans = KMeans(n_clusters=3, random_state = 12)
    kmeans.fit(A)
    kmeans.predict(A)
    # creating a kmeans and fitting it
    df['v_c'] = kmeans.predict(A)
    
    a = pd.get_dummies(df.v_a)
    b = pd.get_dummies(df.d_a)
    c = pd.get_dummies(df.s_d)
    d = pd.get_dummies(df.v_c)
    
    a = a.rename(columns = {0: 'v_a1', 1: 'v_a2', 2:'v_a3'})
    b = b.rename(columns = {0: 'd_a1', 1: 'd_a2', 2:'d_a3'})
    c = c.rename(columns = {0: 's_d1', 1: 's_d2', 2:'s_d3'})
    d = d.rename(columns = {0: 'v_c1', 1: 'v_c2', 2:'v_c3'})
    
    df = pd.concat([df,a,b,c,d], axis = 1)
    
    df = df.drop(columns = ['volatile_acidity', 'citric_acid', 'sugar', 'density', 'alcohol', 'v_a', 'd_a', 's_d', 'v_c'])
    
    df['qual'] = df['quality'] >= 6
    
    df['qual'] = np.where(df['qual'] == True, 1, 0)
    
    df = df.drop(columns = 'quality')
    
    df = df.rename(columns = {'qual': 'is_good'})
    
    return df
    

def knn_model(X_train, y_train, X_val, y_val):
    '''This function takes in train and validate splits and fits on train and predicts on
    train and validate using KNN.
    '''

    import sklearn.preprocessing
    from sklearn.neighbors import KNeighborsClassifier
        
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    y_pred_proba = knn.predict_proba(X_train)
    knn_score_train = knn.score(X_train, y_train)
    
    y_pred = knn.predict(X_val)
    y_pred_proba = knn.predict_proba(X_val)
    knn_score_val = knn.score(X_val, y_val)
    
    return knn_score_train, knn_score_val


def rf_model(X_train, y_train, X_val, y_val):
    '''This function takes in train and validate splits and fits on train and predicts on
    train and validate using Random Forrest.
    '''

    import sklearn.preprocessing
    from sklearn.ensemble import RandomForestClassifier
        
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=200,
                            max_depth=4, 
                            random_state=123)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    y_pred_proba = rf.predict_proba(X_train)
    rf_score_train = rf.score(X_train, y_train)
    
    y_pred = rf.predict(X_val)
    y_pred_proba = rf.predict_proba(X_val)
    rf_score_val = rf.score(X_val, y_val)
    
    return rf_score_train, rf_score_val


def dt_model(X_train, y_train, X_val, y_val):
    '''This function takes in train and validate splits and fits on train and predicts on
    train and validate using Decision Tree.
    '''

    import sklearn.preprocessing
    from sklearn.tree import DecisionTreeClassifier, plot_tree
        
    train_tree = DecisionTreeClassifier(max_depth=7, random_state=77)
    train_tree = train_tree.fit(X_train, y_train)
    y_pred = train_tree.predict(X_train)
    y_pred_proba = train_tree.predict_proba(X_train)
    dt_score_train = train_tree.score(X_train, y_train)
    
    y_pred = train_tree.predict(X_val)
    y_pred_proba = train_tree.predict_proba(X_val)
    dt_score_val = train_tree.score(X_val, y_val)
    
    return dt_score_train, dt_score_val


def dt_model_test(X_test, y_test, X_train, y_train):
    '''This function takes in test splits and predicts on the test data using Decision Tree.
    '''

    import sklearn.preprocessing
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
        
    train_tree = DecisionTreeClassifier(max_depth=7, random_state=77)
    train_tree = train_tree.fit(X_train, y_train)
    
    y_pred = train_tree.predict(X_test)
    y_pred_proba = train_tree.predict_proba(X_test)
    dt_score_test = train_tree.score(X_test, y_test)
    
    ac = accuracy_score(y_test, y_pred)
    print('Accuracy is: ',ac)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm,annot=True)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Fine Wine Decision Tree Model Results') 

    return plt.show()


def acidity_visual(df):
    '''
    This function creates a countplot of the acidity levels and quality of wine.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df['good_or_not'] = df.quality >= 6
    binned = pd.cut(x = df.volatile_acidity, bins = 4)
    binned2 = pd.cut(x = df.citric_acid, bins = 4)

    plt.subplot(1, 2, 1)
    sns.countplot(data = df, x = binned, hue = 'good_or_not', palette=['beige', 'maroon'])
    plt.xlabel('Volatile Acidity')
    plt.ylabel('Bottles of wine')
    plt.xticks([])

    plt.subplot(1, 2, 2)
    sns.countplot(data = df, x = binned2, hue = 'good_or_not', palette=['beige', 'maroon'])
    plt.xlabel('Citric Acid')
    plt.xticks([]) 
    plt.yticks([])
    a = plt.gca()
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)

    plt.suptitle('The lower the acidity the better')
    return plt.show()



def sugar_visual(df):
    '''
    This function creates a countplot of the sugar levels and quality of wine.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df['good_or_not'] = df.quality >= 6
    binned = pd.cut(x = df.sugar, bins = 4)
    sns.countplot(data = df, x = binned, hue = 'good_or_not', palette=['beige', 'maroon'])
    plt.title('Lower sugar levels produce higher quality wine')
    plt.xlabel('Sugar')
    plt.ylabel('Bottles of wine')
    plt.show()
    
    
    
def density_visual(df):
    '''
    This function creates a countplot of the density levels and quality of wine.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df['good_or_not'] = df.quality >= 6
    binned = pd.cut(x = df.density, bins = 4)
    sns.countplot(data = df, x = binned, hue = 'good_or_not', palette=['beige', 'maroon'])
    plt.title('Lower to moderate density produces higher quality wine')
    plt.xlabel('Density Level')
    plt.ylabel('Bottles of wine')
    plt.show()
    
    
    
def alcohol_visual(df):
    '''
    This function creates a countplot of the density levels and quality of wine.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df['good_or_not'] = df.quality >= 6
    binned = pd.cut(x = df.alcohol, bins = 4)
    sns.countplot(data = df, x = binned, hue = 'good_or_not', palette=['beige', 'maroon'])
    plt.title('Balancing alcohol produces quality wine')
    plt.xlabel('Alcohol')
    plt.ylabel('Bottles of wine')
    plt.show()
    