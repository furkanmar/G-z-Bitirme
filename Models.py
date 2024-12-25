import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
import seaborn as sns
import matplotlib.pyplot as plt

#Read data into dataframe
df = pd.read_csv('synthetic_data.csv')


# Target parameter: "Mutation (%)" 
target = 'Mutation (%)'
X = df.drop(columns=[target])
y = df[target]


# Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Corelation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Korelasyon Matrisi")
plt.show()


# Histograms
df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()


# Random Forest
param_dist = {
    'feature_selection__k': [5, 10, 'all'],
    'rf__n_estimators': [ 500, 1000],
    'rf__max_depth': [None, 5, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}


pipeline = Pipeline([

    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  
    ('rf', RandomForestRegressor(random_state=42))
])

# Hiperparameter optimization for Random Forest 
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=5, 
    scoring='r2', 
    random_state=42, 
    n_jobs=-1
)
random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred)
rf_mape = mean_absolute_percentage_error(y_test, y_pred)
rf_ev = explained_variance_score(y_test, y_pred)

print("Best parameters:", random_search.best_params_)
print(f"R2 score: {rf_r2:.4f}")
print(f"MAPE: {rf_mape:.4f}")
print(f"Explained Variance: {rf_ev:.4f}")



# Support Vector Regressor
param_dist = {
    'feature_selection__k': [5, 10, 'all'],
    'svr__C': [0.1, 1, 10, 100, 1000],
    'svr__epsilon': [0.001, 0.01, 0.1, 1],
    'svr__kernel': ['rbf', 'poly', 'linear'],
    'svr__gamma': ['scale', 0.001, 0.01, 0.1, 1], 
}


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('feature_selection', SelectKBest(score_func=f_regression)),
    ('svr', SVR())
])

# Hiperparameter optimization for Support Vector Regressor
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=5, 
    scoring='r2', 
    random_state=42, 
    n_jobs=-1
)

random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_


y_pred = best_model.predict(X_test)
svr_r2 = r2_score(y_test, y_pred)
svr_mape = mean_absolute_percentage_error(y_test, y_pred)
svr_ev = explained_variance_score(y_test, y_pred)

print("Best parameters:", random_search.best_params_)
print(f"R2 score: {svr_r2:.4f}")
print(f"MAPE: {svr_mape:.4f}")
print(f"Explained Variance: {svr_ev:.4f}")




# Electic Net

param_dist = {
    'feature_selection__k': [5, 10, 'all'],  
    'elastic__alpha': np.logspace(-2, 1, 10),  
    'elastic__l1_ratio': np.linspace(0.1, 0.9, 9),

}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ('elastic', ElasticNet(random_state=42, max_iter=10000))

])

# Hiperparameter optimization for Elastic Net
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=50,             
    cv=10, 
    scoring='r2', 
    random_state=42, 
    n_jobs=-1,ww
    verbose=2
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)


en_r2 = r2_score(y_test, y_pred)
en_mape = mean_absolute_percentage_error(y_test, y_pred)
en_ev = explained_variance_score(y_test, y_pred)

print("Best Parameter:", random_search.best_params_)
print(f"R2 Score: {en_r2:.4f}")
print(f"MAPE: {en_mape:.4f}")
print(f"Explained Variance: {en_ev:.4f}")




data = {
    "Model": ["Random Forest", "Support Vector Regressor", "ElasticNet"],
    "R2 Score": [rf_r2, svr_r2, en_r2],
    "MAPE": [rf_mape, svr_mape, en_mape],
    "Explained Variance": [rf_ev, svr_ev, en_ev]
}


df = pd.DataFrame(data)
df.to_csv("model_scores_v1.csv", index=False)


plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["R2 Score"], alpha=0.7, label=("R2 Score"), color='blue')


plt.title("Model Evaluation R2_Score")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.ylim(0.9,1)
plt.xticks(rotation=45)
plt.legend()


plt.savefig("model_scores_plot_v1.png")
plt.show()
