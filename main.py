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
    YEAR_LIST = np.arange(1939, 2015)

    # Number of males and females
    number_of_males = 0
    number_of_females = 0

    # Create list of males and females
    number_of_males_list = np.zeros(len(YEAR_LIST))
    number_of_females_list = np.zeros(len(YEAR_LIST))

    # Gross income
    total_gross_male = 0
    total_gross_female = 0

    for row in csv.itertuples():
        if row.Lead == 'Male':
            number_of_males_list[int(row.Year) - 1940] += 1
            number_of_males += 1
            total_gross_male += int(row.Gross)
        else:
            number_of_females_list[int(row.Year) - 1940] += 1
            number_of_females += 1
            total_gross_female += int(row.Gross)
        
    # Plot the results    
    plt.plot(YEAR_LIST, number_of_males_list, label = 'male')
    plt.plot(YEAR_LIST, number_of_females_list, label = 'female')
    plt.xlabel('year')
    plt.ylabel('number of lead actors')
    plt.legend()
    plt.show()
        
    print(f'Number of males: {number_of_males}' )
    print(f'Number of females: {number_of_females}')

    print(f'Average gross for males: {total_gross_male / number_of_males}')
    print(f'Average gross for females: {total_gross_female / number_of_females}')


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
    model = skl_da.QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
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
    # statistics()
    naive_classifier()
    LDA()
    QDA()
    kNN(k=5)
    forest()
    bagging()
    trees()
    # find_optimal_kNN()
    # cross_validation()
    create_prediction()

    

if __name__ == '__main__':
    main()



