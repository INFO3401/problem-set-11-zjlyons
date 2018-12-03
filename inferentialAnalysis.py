import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df


def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv])
    print(ttest)
    
def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print(aov_table)

#Monday - Problem 1
#Problem A
#IV = year in school (categorical)
#DV = GPA (continuous)
#Test: T-test

#Problem B
#IV = time (continuous)
#DV = inches of snow (continuous)
#Test: Generalized Regression

#Problem C
#IV = Season (categorical)
#DV = Amount of hikers (discrete)
#Test: T-test

#Problem D
#IV = home state (categorical)
#DV = highest degree achieved (categorical)
#Test: Chi-Squared
#Monday - Problem 2 

rawData, df = generateDataset('simpsons_paradox.csv')
df['total'] = df['Admitted'] + df['Rejected']
df['AcceptanceRate'] = df['Admitted']/df['total']

print("Does gender correlate with admissions?")
men = df[(df['Gender']=='Male')]
women = df[(df['Gender']=='Female')]
runTTest(men, women, 'Admitted')

print('Does department correlate with admissions?')
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData,simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData,moreComplex)

#Results from our first set of tests would indicate that gender has a significant
#influence on admissions, but department does not. These results would indicate
#that the admisssions process is biased based on gender, given that gender has a 
#significance level of 0.0851 in our ANOVA test. Similar results in the T-Test 
#verify this claim. 


#Monday - Problem 3
#Not all departments have an equal amount of applicants. Therefore, we can solve our data problem by creating a new column within our dataframe that refelcts the percentage of applicants that made it in to each department. Code to do so is within problem 2.

print("Does gender correlate with admissions?")
men = df[(df['Gender']=='Male')]
women = df[(df['Gender']=='Female')]
runTTest(men, women, 'AcceptanceRate')

print('Does department correlate with admissions?')
percentageFormula = 'AcceptanceRate ~ C(Department)'
runAnova(df,percentageFormula)

print("Do gender and department correlate with admissions?")
moreComplexPercent = 'AcceptanceRate ~ C(Department) + C(Gender)'
runAnova(df,moreComplexPercent)

#In our two sets of tests, we see a significant decrease in the evidence that gender
#influences the the admissions process, once we change our dependent variables to 
#ratios. Between my Problem 2 and Problem 3 tests, the p-value of the gender 
#variable actually increased. They remain roughly the same when measuring the 
#influence that department has on admissions. 

#Problem #4 Will be located with all of my other Altair Visualizations in a seperate .ipynb notebook


