import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

csv = pd.read_csv('train.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

def main():
    LEAD = csv['Lead']
    YEAR = csv['Year']
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
    

def DiscriminantAnalysis():
    n = 0







if __name__ == '__main__':
    main()



