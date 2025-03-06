#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the necessary libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import sweetviz as sv
import warnings
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:





# In[2]:


df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")
print("Full train dataset shape is {}".format(df.shape))
df_org = df.copy()


# In[3]:


df.info()


# In[4]:


df.isnull()


# In[5]:


# Checking descriptive statistics
df.describe().T


# In[6]:


## Checking percentage of missing values
missing_info= round(df.isna().sum() * 100/df.shape[0], 2)
missing_info[missing_info > 0].sort_values(ascending= False)


# In[7]:


# Getting column names having missing values
missing_val_cols= missing_info[missing_info > 0].sort_values(ascending= False).index
missing_val_cols


# In[8]:


# Checking unique values in these columns
for col in missing_val_cols:
    print('\nColumn Name:',col)
    print(df[col].value_counts(dropna= False))


# In[9]:


# Replacing NaN with 'Not Present' for below columns
valid_nan_cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
df[valid_nan_cols]= df[valid_nan_cols].fillna('Not Present')


# In[10]:


# Checking percentage of missing values again
missing_info= round(df.isna().sum() * 100/df.shape[0], 2)
missing_info[missing_info > 0].sort_values(ascending= False)


# In[11]:


# Checking if there is any relation between GarageYrBlt and GarageType
df[df.GarageYrBlt.isna()]['GarageType'].value_counts(normalize= True)


# There are a total of 19 columns with missing values. Notably, PoolQC, MiscFeature, Alley, Fence, and FireplaceQu exhibit a particularly high percentage of missing values. It is crucial to examine these columns individually to determine whether these missing values are indeed indicative of actual data gaps or if they hold some meaningful information. Once we identify the nature of these missing values, we can proceed with the appropriate imputation technique. One approach is to utilize business knowledge for imputation, where NaN values can be replaced with values derived from relevant business logic. Alternatively, statistical imputation methods can be employed after performing a train-test split, allowing missing values to be filled based on statistical techniques.

# In[12]:


# Getting column names having missing values
missing_val_cols= missing_info[missing_info > 0].sort_values(ascending= False).index
missing_val_cols


# In[13]:


# Checking unique values in these columns
for col in missing_val_cols:
    print('\nColumn Name:',col)
    print(df[col].value_counts(dropna= False))


# In[14]:


# Replacing NaN with 'Not Present' for below columns
valid_nan_cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
df[valid_nan_cols]= df[valid_nan_cols].fillna('Not Present')


# In[15]:


# Checking percentage of missing values again
missing_info= round(df.isna().sum() * 100/df.shape[0], 2)
missing_info[missing_info > 0].sort_values(ascending= False)


# In[16]:


# Checking if there is any relation between GarageYrBlt and GarageType
df[df.GarageYrBlt.isna()]['GarageType'].value_counts(normalize= True)


# Initially, both GarageYrBlt and GarageType had a missing value percentage of 5.55%. However, after replacing the NaN values of GarageType with 'Not Available', we observed that the GarageYrBlt values were only NaN for those instances where GarageType was labeled as 'Not Available'. This indicates that if a garage is not available, there will be no corresponding value for 'GarageYrBlt'. As a result, it is reasonable to safely replace the NaN values of GarageYrBlt with 0.

# In[17]:


# Imputing missing values of GarageYrBlt column
df['GarageYrBlt']= df['GarageYrBlt'].fillna(0)


# In[18]:


# Changing data type of MSSubClass. MSSubClass: "identifies the type of dwelling involved in the sale", 
#is a categorical variable, but it's appearing as a numeric variable.
df['MSSubClass']= df['MSSubClass'].astype('object')


# In[19]:


df.info()


# In[20]:


# Running SweetViz AutoEDA
sv_report= sv.analyze(df)
sv_report.show_notebook()


# In[21]:


# Plotting numeric variables against SalePrice

numeric_cols= ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea',
'BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea']

sns.pairplot(df, x_vars=['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(df, x_vars=['YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','LotFrontage'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(df, x_vars=['WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})


# In[22]:


# Box plot of catego

cat_cols= ['OverallQual','GarageCars','ExterQual','BsmtQual','KitchenQual','FullBath','GarageFinish','FireplaceQu','Foundation','GarageType','Fireplaces','BsmtFinType1','HeatingQC']

plt.figure(figsize=[18, 40])

for i, col in enumerate(cat_cols, 1):
    plt.subplot(7,2,i)
    title_text= f'Box plot {col} vs cnt'
    x_label= f'{col}'
    fig= sns.boxplot(data= df, x= col, y= 'SalePrice', palette= 'Greens')
    fig.set_title(title_text, fontdict= { 'fontsize': 18, 'color': 'Green'})
    fig.set_xlabel(x_label, fontdict= {'fontsize': 12, 'color': 'Brown'})
plt.show()


# In[23]:


plt.figure(figsize=[17,7])
sns.boxplot(data= df, x= 'Neighborhood', y= 'SalePrice', palette= 'Greens')
plt.show()


# In[24]:


plt.figure(figsize=(24, 12))
sns.heatmap(df.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)


# In[25]:


obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[26]:


df.info()


# In[27]:


unique_values = []
for col in object_cols:
    unique_values.append(df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[28]:


plt.figure(figsize=(30, 56))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = df[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1


# In[29]:


sns.boxenplot(x=df["SalePrice"])


# In[30]:


sns.lmplot(x = "SalePrice", y = "GarageArea",
           hue = "LotShape", data = df, legend = True,aspect=20/12)


# In[31]:


sns.boxenplot(data=df, x="SalePrice", y="MSZoning", scale="linear")


# In[32]:


g = sns.JointGrid(data=df, x="SalePrice", y="LotArea")
g.plot_joint(sns.scatterplot, s=100, alpha=.5)
g.plot_marginals(sns.histplot, kde=True)


# In[33]:


sns.catplot(data=df, x="HouseStyle", y="SalePrice", aspect= 20/18)


# In[34]:


sns.catplot(data=df, x="SalePrice", y="Neighborhood", kind="box")


# In[35]:


sns.catplot(data=df, x="SalePrice", y="BldgType", kind="box")


# In[36]:


#Scatter plot to find relation between GarageArea and SalePrice
sns.set_style('darkgrid')

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='GarageArea', y='SalePrice', s=80, alpha=0.7, color="#008080")
plt.title("Sale Price vs. Garage Area", fontsize=18)
plt.xlabel("Garage Area", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[37]:


sns.lineplot(data=df, x="SalePrice", y="GarageArea")


# In[38]:


sns.distplot((df['SalePrice']))
plt.show()


# In[39]:


# Distplot of log transformed SalePrice to reduce skewness.
sns.distplot(np.log(df['SalePrice']))
plt.show()


# In[ ]:





# In[40]:


# Transforming 'SalePrice'
df['SalePrice_log_trans']= np.log(df['SalePrice'])


# In[41]:


# Dropping ID Column and SalePrice
df.drop(['SalePrice','Id'], axis=1, inplace= True)
df.shape


# In[42]:


df.shape


# In[43]:


# Train-Test Split
y= df['SalePrice_log_trans']
X= df.drop('SalePrice_log_trans', axis= 1)

X_train, X_test, y_train, y_test= train_test_split(X, y, train_size= .7, random_state= 42)
df.shape


# In[44]:


# Getting index values of train test dataset
train_index= X_train.index
test_index= X_test.index


# In[45]:


# Performing Statistical Imputation for missing values in LotFrontage, MasVnrArea, MasVnrType, Electrical columns

df['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)
df['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)

df['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)
df['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)

df['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)
df['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)

df['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)
df['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)


# In[46]:


# Getting object and numeric type columns
housing_cat= df.select_dtypes(include= 'object')
housing_num= df.select_dtypes(exclude= 'object')
housing_cat.describe()


# In[47]:


# 'Street','Utilities', 'CentralAir' have 2 unique data, so we are encoding with 0 and 1
df['Street']= df.Street.map(lambda x: 1 if x== 'Pave' else 0)
df['Utilities']= df.Utilities.map(lambda x: 1 if x== 'AllPub' else 0)
df['CentralAir']= df.CentralAir.map(lambda x: 1 if x== 'Y' else 0)


# In[48]:


# Performing get_dummies
cat_cols= housing_cat.columns.tolist()
done_encoding= ['Street','Utilities', 'CentralAir']
cat_cols= [col for col in cat_cols if col not in done_encoding]
dummies= pd.get_dummies(df[cat_cols], drop_first=True)


# In[49]:


# Checking all dummies
dummies.head()


# In[50]:


# Concatinating dummies with housing_df dataframe and droping original features
print('housing_df before droping original valiables', df.shape)
print('shape of dummies dataframe', dummies.shape)
df.drop(cat_cols, axis=1, inplace= True)
df= pd.concat([df, dummies], axis= 1)
print('final shape of housing_df', df.shape)


# In[51]:


X_train=X_train.drop('Utilities',axis=1)


# In[52]:


# Re-constructing Train-test data
X_train= df.iloc[train_index, :].drop('SalePrice_log_trans', axis= 1)
y_train= df.iloc[train_index, :]['SalePrice_log_trans']
X_test= df.iloc[test_index, :].drop('SalePrice_log_trans', axis= 1)
y_test= df.iloc[test_index, :]['SalePrice_log_trans']


# In[53]:


# Performing scaling of numeric columns in training and test dataset using RobustScaler
num_cols= housing_num.columns.tolist()
num_cols.remove('SalePrice_log_trans')
scaler= RobustScaler(quantile_range=(2, 98))
scaler.fit(X_train[num_cols])
X_train[num_cols]= scaler.transform(X_train[num_cols])
X_test[num_cols]= scaler.transform(X_test[num_cols])


# In[54]:


# Checking scaled features
X_train[num_cols].head()


# In[55]:


var_t= VarianceThreshold(threshold= .003)
variance_thresh= var_t.fit(X_train)
col_ind= var_t.get_support()

# Below columns have very low variance
X_train.loc[:, ~col_ind].columns


# In[56]:


# Checking number of apperance of one of the attributes/categorical value in dataset
df_org.Functional.value_counts()


# In[57]:


# Removing above columns from train and test dataset
X_train= X_train.loc[:, col_ind]
X_test= X_test.loc[:, col_ind]


# In[58]:


# Checking shape of final training dataset
X_train.shape


# In[59]:


# Selecting few values for alpha
range1= [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
range2= list(range(2, 1001))
range1.extend(range2)
params_grid= {'alpha': range1}


# In[60]:


# Applying Ridge and performing GridSearchCV to find optimal value of alpha (lambda)

ridge= Ridge(random_state= 42)
gcv_ridge= GridSearchCV(estimator= ridge, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)      
gcv_ridge.fit(X_train, y_train)


# In[61]:


# Checking best estimator 
gcv_ridge.best_estimator_


# In[62]:


# Checking best MAE
gcv_ridge.best_score_


# In[63]:


# Checking best MAE
gcv_ridge.best_score_


# In[64]:


# Fitting model using best_estimator_
ridge_model= gcv_ridge.best_estimator_
ridge_model.fit(X_train, y_train)


# In[65]:


models = ["RidgeCV", "LassoRegression", "KNN_Regressor","CrossValidationKNN_Regressor", "DecisonTreeRegressor", "GradientBoostingRegressor", "RandomForestRegressor", "AdaBoostRegressor", "CatBoostRegressor", "PassiveAggressiveRegressor"]


# In[66]:


r2_test = []
MSE_test = []
RMSE_test = []
MAE_test = []


# In[67]:


# Evaluating on training dataset
y_train_pred= ridge_model.predict(X_train)
print(y_train_pred)
print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[68]:


for i in y_train_pred:
    print(np.exp(i))


# In[69]:


# Evaluating on testing dataset
y_test_pred= ridge_model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))

r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[70]:


for i in y_test_pred:
    print(np.exp(i))


# In[71]:


# Ridge coefficients
ridge_model.coef_


# In[72]:


# Ridge intercept
ridge_model.intercept_


# In[73]:


# Top 10 features with double the value of optimal alpha in Ridge
ridge_coef= pd.Series(ridge_model.coef_, index= X_train.columns)
top_25_ridge=  ridge_coef[abs(ridge_coef).nlargest(25).index]
top_25_ridge


# Lasso Regression

# In[74]:


# Applying Lasso and performing GridSearchCV to find optimal value of alpha (lambda)

params_grid= {'alpha': range1}
lasso= Lasso(random_state= 42)
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train)         


# In[75]:


# Checking best estimator 
lasso_gcv.best_estimator_


# In[76]:


# Checking best MAE
lasso_gcv.best_score_


# In[77]:


range3= [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, .001]
params_grid= {'alpha': range3}
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train)     


# In[78]:


# Checking best estimator 
lasso_gcv.best_estimator_


# In[79]:


# Fitting model using best_estimator_
lasso_model= lasso_gcv.best_estimator_
lasso_model.fit(X_train, y_train)


# In[80]:


# Evaluating on training dataset
y_train_pred= lasso_model.predict(X_train)
print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[81]:


for i in y_train_pred:
    print(np.exp(i))


# In[82]:


# Evaluating on testing dataset
y_test_pred= lasso_model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[83]:


for i in y_test_pred:
    print(np.exp(i))


# In[84]:


# Checking no. of features in Ridge and Lasso models
lasso_coef= pd.Series(lasso_model.coef_, index= X_train.columns)
selected_features= len(lasso_coef[lasso_coef != 0])
print('Features selected by Lasso:', selected_features)
print('Features present in Ridge:', X_train.shape[1])


# In[85]:


# Lasso intercept
lasso_model.intercept_


# In[86]:


# Top 25 features with coefficients in Lasso model
top25_features_lasso=  lasso_coef[abs(lasso_coef[lasso_coef != 0]).nlargest(25).index]
top25_features_lasso


# In[87]:


# Ploting top 25 features
plt.figure(figsize= (7, 5))
top25_features_lasso.plot.barh(color= (top25_features_lasso > 0).map({True: 'g', False: 'r'}))
plt.show()


# In[88]:


#KNN Regressor
uniform  = []
distance = []
r = range (1,21,2)

for k in r:
    
    # Euclidan, 'straight' distance
    model = KNeighborsRegressor(n_neighbors = k, weights='uniform')
    model.fit(X_train.values, y_train.values)
    uniform.append(model.score(X_test.values,y_test.values))

    # Distance is inversely proportional (to lessen the weight of outliers)
    model = KNeighborsRegressor(n_neighbors = k, weights='distance') 
    model.fit(X_train.values, y_train.values)
    distance.append(model.score(X_test.values,y_test.values))

uniform = np.array(uniform)
distance = np.array(distance)

plt.rcParams['figure.figsize'] = [10, 3]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
plt.plot(r,uniform,label='uniform',color='blue')
plt.plot(r,distance,label='distance',color='red')
plt.legend()
plt.gca().set_xticks(r)
plt.show()


# In[89]:


y_test_pred= model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_train, y_train_pred))
MAE_test.append(mean_absolute_error(y_train, y_train_pred))
MSE_test.append(mean_squared_error(y_train, y_train_pred))
RMSE_test.append(mean_squared_error(y_train, y_train_pred)**.5)


# In[90]:


for i in y_test_pred:
    print(np.exp(i))


# In[91]:


#CrossValidation for KNN Regressor
pd.DataFrame({"k" : r, "uniform" : uniform, "distance" : distance})


# In[92]:


params = {'n_neighbors':range(1,21,2),'weights':['uniform','distance']}
model = GridSearchCV(KNeighborsRegressor(), params, cv=5)
model.fit(X_train.values,y_train.values)
model.best_params_


# In[93]:


model.score(X_test.values,y_test.values)


# In[94]:


y_test_pred= model.predict(X_test)
print(y_test_pred)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[95]:


for i in y_test_pred:
    print(np.exp(i))


# In[96]:


model = DecisionTreeRegressor()

model.fit(X_train.values,y_train.values)


# In[97]:


y_test_pred= model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[98]:


for i in y_test_pred:
    print(np.exp(i))


# In[99]:


reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
GradientBoostingRegressor(random_state=0)


# In[100]:


y_test_pred= reg.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[101]:


for i in y_test_pred:
    print(np.exp(i))


# In[102]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)


# In[103]:


y_test_pred= regr.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[104]:


for i in y_test_pred:
    print(np.exp(i))


# In[105]:


from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
AdaBoostRegressor(n_estimators=100, random_state=0)


# In[106]:


y_test_pred= regr.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[107]:


for i in y_test_pred:
    print(np.exp(i))


# In[109]:


from catboost import CatBoostRegressor
cb_reg_1 = CatBoostRegressor( random_seed=13, verbose=200)
cb_reg_1.fit(X_train, y_train)


# In[110]:


y_test_pred= cb_reg_1.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[111]:


for i in y_test_pred:
    print(np.exp(i))


# In[112]:


df.to_csv('train1')


# In[113]:


from sklearn.linear_model import PassiveAggressiveRegressor
regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)
regr.fit(X_train, y_train)
PassiveAggressiveRegressor(max_iter=100, random_state=0)


# In[114]:


y_test_pred= regr.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))
r2_test.append(r2_score(y_test, y_test_pred))
MAE_test.append(mean_absolute_error(y_test, y_test_pred))
MSE_test.append(mean_squared_error(y_test, y_test_pred))
RMSE_test.append(mean_squared_error(y_test, y_test_pred)**.5)


# In[ ]:


for i in y_test_pred:
      print(np.exp(i))


# In[ ]:


print(len(r2_test))


# In[ ]:


print(r2_test)


# In[115]:


plt.figure(figsize=(10,6))
plt.title('Comparision chart for R2 Score')
plt.xticks(rotation=90)
sns.barplot(x=models,y=r2_test)


# In[116]:


plt.figure(figsize=(10,6))
plt.title('Comparision chart for Mean Absolute Error')
plt.xticks(rotation=90)
sns.barplot(x=models,y=MAE_test)


# In[117]:


plt.figure(figsize=(10,6))
plt.title('Comparision chart for Mean Squared Error')
plt.xticks(rotation=90)
sns.barplot(x=models,y=MSE_test)


# In[118]:


plt.figure(figsize=(10,6))
plt.title('Comparision chart for Root Mean Squared Error')
plt.xticks(rotation=90)
sns.barplot(x=models,y=RMSE_test)


# In[ ]:




