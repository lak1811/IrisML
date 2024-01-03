#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Assigning columns and making a dataframe with all the necessary data
kolonner = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', names=kolonner)

#Making histograms for all the columns, except species. This is to present the data i am going to analyze
for label in kolonner[:-1]:
    plt.hist(df[df['species'] == 'Iris-setosa'][label], color='blue', density=True, alpha=0.7, label='setosa')
    plt.hist(df[df['species'] == 'Iris-virginica'][label], color='red', density=True, alpha=0.7, label='virginica')
    plt.hist(df[df['species'] == 'Iris-versicolor'][label], color='green', density=True, alpha=0.7, label='versicolour')
    plt.ylabel(label)
    plt.xlabel('Amount')
    plt.legend()
    plt.show()
    

#Make X and Y for later testing and to create testing/training sets. I have to have a DF with only the species and one with the other data.
X = df.drop('species', axis=1)
y = df['species']

#Making the training sets
X_train, test_X, y_train, test_y = train_test_split(X, y, test_size=0.7, random_state=42)

#Making and training different kind of models to see the results and accuracies i could get

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_predict_knn = knn_model.predict(test_X)

forestmodel=RandomForestClassifier(n_estimators=100)
forestmodel.fit(X_train, y_train)
y_predforest = forestmodel.predict(test_X)


dtmodel=DecisionTreeClassifier()
dtmodel.fit(X_train,y_train)
y_preddt=dtmodel.predict(test_X)


svc_model=SVC(probability=True)
svc_model.fit(X_train,y_train)
ypredsvc=svc_model.predict(test_X)

#Wanted to see the accuracy that could be reached with a votingclassifier (soft)
voting_clf = VotingClassifier(estimators=[('svc', svc_model), ('forestclassifier', forestmodel), ('DecisionTree', dtmodel),('KNN', knn_model)], voting='soft')
voting_clf.fit(X_train, y_train)
y_predvoting=voting_clf.predict(test_X)


#Printing a report of all the classifiers

print("----------KNeighbors classifier----------")
print(classification_report(test_y, y_predict_knn))

print("---------- Random Forest classifier----------")
print(classification_report(test_y,y_predforest))

print("----------Decision Tree classifier----------")
print(classification_report(test_y,y_preddt))

print("----------SVC classifier----------")
print(classification_report(test_y,ypredsvc))

print ("---------------Voting classifier-----------")
print(classification_report(test_y,y_predvoting))


#Making a testing dataset. If my datasets are correct, they should predict "iris-setosa, iris-versicolor, iris-setosa"
new_flower_measurements = pd.DataFrame({
    'sepal_length': [5.1, 6.2, 4.8],
    'sepal_width': [3.5, 2.9, 3.4],
    'petal_length': [1.4, 4.3, 1.9],
    'petal_width': [0.2, 1.3, 0.4]
})


#Predicting
knn_predictions = knn_model.predict(new_flower_measurements)


forest_predictions = forestmodel.predict(new_flower_measurements)

dt_predictions = dtmodel.predict(new_flower_measurements)


svcpredictions=svc_model.predict(new_flower_measurements)

votingpredictor=voting_clf.predict(new_flower_measurements)


# Displaying the predictions
print (f"KNN predictor: {knn_predictions}")
print (f"Random Forest predictor: {forest_predictions}")
print(f"Decision Tree predictor: {dt_predictions}")
print (f"SVC predictor: {svcpredictions}")
print (f"Voting predictor: {votingpredictor}")


