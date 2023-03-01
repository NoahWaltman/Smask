import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv_module
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.ensemble as skl_en
import sklearn.tree as skl_tree
import sklearn.metrics as skl_met
from operator import add, truediv

# 1. Use a train and validation set
#
# 2. Use Accuracy. (Maybe switch to cross validation later)
#
# 3.  
# 
# 4. 
# 
# 5. Use all coloumns except ('Lead', 'Number words male') for X and 
#    use only coloumn ('Lead') for y. 
#    'Number words female' and 'Number words male' are colinear 
#    so we only use one of them.
#


# Pre-process data
csv = pd.read_csv('train.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
X = csv.drop(columns=['Lead', 'Number words male'])
y = csv['Lead']
X_train, X_val, y_train, y_val = skl_ms.train_test_split(X, y, test_size = 0.3, random_state = 1)


def statistics():

    # Create list of males and females
    male_dict = {}
    female_dict = {}

    male_gross = []
    female_gross = []

    for index, row in csv.iterrows():
        year = int(row['Year'])
        num_male_actors = int(row['Number of male actors'])
        num_female_actors = int(row['Number of female actors'])
        
        male_words = int(row['Number words male'])
        female_words = int(row['Number words female'])
        gross = int(row['Gross'])
        
        if male_words > female_words:
            male_gross.append(gross)
        else:
            female_gross.append(gross)
        

        if year in male_dict:
            male_dict[year] += num_male_actors
        else:
            male_dict[year] = num_male_actors
        
        if year in female_dict:
            female_dict[year] += num_female_actors
        else:
            female_dict[year] = num_female_actors


    male_list = sorted(list(male_dict.values()))
    female_list = sorted(list(female_dict.values()))
    total_list = list(map(add, male_list, female_list))
    year = sorted(list(male_dict.keys()))
    percent_list = list(map(truediv, male_list, total_list))
    

    # Plot the results    
    plt.scatter(year, percent_list, s=8)
    m, b = np.polyfit(year, percent_list, 1)
    plt.plot(year, m*np.array(year) + b, color='red')
    plt.ylim(0,1)
    plt.xlabel('year')
    plt.ylabel('percentage of male actors')
    plt.show()
    plt.boxplot((male_gross, female_gross), sym='')
    plt.xticks(np.arange(2)+1, ('male', 'female'))
    plt.ylabel('gross income')
    plt.show()
        


def naive_classifier():
    prediction = np.full(len(X_val), 'Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for Naive Classifier: {round(misclassification,3)}')


def LDA():
    model = skl_da.LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for LDA: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')


def QDA():
    model = skl_da.QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for QDA: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')
    return model


def kNN(k, test=False):
    model = skl_nb.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    if test:
        return misclassification
    print(f'Error for kNN: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')
    


def find_optimal_kNN():
    N = 50
    errors = np.zeros(N)
    for k in range(N):
        errors[k] = kNN(k+1, test=True)
    min_error = np.min(errors)
    min_k = np.argmin(errors)
    print(min_k+1, round(min_error, 3))


def forest():
    model = skl_en.RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for forest: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')


def bagging():
    model = skl_en.BaggingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for bagging: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')


def trees():
    model = skl_tree.DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    f1_score = skl_met.f1_score(prediction, y_val, pos_label='Male')
    misclassification = np.mean(prediction != y_val)
    print(f'Error for trees: {round(misclassification,3)}')
    print(f'f1 score for LDA: {f1_score}')


def cross_validation():
    models = []
    models.append(skl_da.LinearDiscriminantAnalysis())
    models.append(skl_da.QuadraticDiscriminantAnalysis())
    models.append(skl_nb.KNeighborsClassifier(n_neighbors=5))
    models.append(skl_en.RandomForestClassifier())
    models.append(skl_en.BaggingClassifier())
    models.append(skl_tree.DecisionTreeClassifier())

    n_fold = 10
    misclassification = np.zeros((n_fold, len(models)))
    cv = skl_ms.KFold(n_splits=n_fold, random_state = 1, shuffle=True)

    for i, (train_index, val_index) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index] 
        for m in range(len(models)):
            model = models[m]
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
            misclassification[i, m] = np.mean(prediction != y_val)

    for m in range(len(models)):
        print(f'Average missclasification for {models[m]}: {np.mean(misclassification[m])}')


    plt.boxplot(misclassification)
    plt.title('Cross validation error for different methods')
    plt.xticks(np.arange(6)+1, ('LDA', 'QDA', 'kNN', 'Forest', 'Bagging', 'Trees'))
    plt.ylabel('validation error')
    plt.show()


def create_prediction():
    model = QDA()
    test_csv = pd.read_csv('test.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
    X_test = test_csv.drop(columns=['Number words male'])
    prediction = model.predict(X_test)
    prediction_list = []
    for p in prediction:
        if p == 'Male':
            prediction_list.append(0)
        else:
            prediction_list.append(1)

    with open('predictions.csv', 'w', newline='') as file:
        writer = csv_module.writer(file)
        writer.writerow(prediction_list)
     
 


def main():
    statistics()
    '''naive_classifier()
    LDA()
    QDA()
    kNN(k=5)
    forest()
    bagging()
    trees()
    # find_optimal_kNN()
    # cross_validation()
    create_prediction()'''

    

if __name__ == '__main__':
    main()



