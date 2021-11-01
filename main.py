import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle

df = pd.read_csv('./Data/winequality_red.csv')
print(df.head())

x = df.drop(columns=['quality'])
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# print(X_scaled)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=355)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

print("The Testing accuracy score is : ", accuracy_score(y_test, y_pred))
print("Confusion Matrix : ", confusion_matrix(y_test, y_pred))
print("Classification Report : ", classification_report(y_test, y_pred))

param_dist = {
    "n_estimators": range(100, 3000, 50),
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 51, 1),
    'min_samples_leaf': range(1, 51, 1),
    'min_samples_split': range(2, 51, 1),
    'max_features': ['auto', 'log2']
}
randomized_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=300, cv=7, random_state=160, n_jobs=-1, verbose=3)
randomized_search.fit(x_train, y_train)

print("After RandomizedSearch CV)")
print('Best Parameters : ', randomized_search.best_params_)
rfc_best = randomized_search.best_estimator_
print(f'rf_best:{rfc_best}')
y_pre = rfc_best.predict(x_test)
print("Accuracy Score :- ", accuracy_score(y_test, y_pre))
print("Confusion Matrix :- ", confusion_matrix(y_test, y_pre))
print("Classification Report :- ", classification_report(y_test, y_pre))


# with open('Final_ModelForPrediction.pkl', 'wb') as f:
#     pickle.dump(rfc_best,f)
# pickle.dump(scaler, open('scaler_model.pkl', 'wb'))

loaded_model = pickle.load(open('Final_ModelForPrediction.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
d=scaler_model.transform([[7, 0.8, 0.5, 12, 0.4, 50, 200, 0.9970, 3, 1.2, 13]])
pred=loaded_model.predict(d)
print('This data belongs to class :', pred[0])

