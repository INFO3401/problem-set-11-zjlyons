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
    
rawData, df = generateDataset('simpsons_paradox.csv') 

print("Does gender correlate with admissions?")
men = df[(df['Gender']=='Male')]
women = df[(df['Gender']=='Male')]
runTTest(men, women, 'Admitted')

print('Does department correlate with admissions?')
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData,simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData,moreComplex)


#Monday - Problem 1