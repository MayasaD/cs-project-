
# coding: utf-8

# In[2]:

from __future__ import print_function, division
#importing the modules that we will use

import pandas as pd
import numpy as np
import random
import thinkstats2
import thinkplot
import nsfg

from scipy import stats


# In[23]:

# The First Part 
# In this part we will get the access to the data 
data = pd.read_csv("Teams.csv") # and then we get the columns that work on
wins = data.W
runs = data.R
print (wins)
print (runs)


# In[26]:

wins_mean = wins.mean() # the mean of the wins
wins_std = wins.std() # the standard deviation of the wins.

print (wins_mean)
print (wins_std)

runs_mean = runs.mean()# the mean of the runs
runs_std = runs.std() # the standard deviation of the runs

print (runs_mean)
print (runs_std)


# In[27]:

# Building the histograms of both the wins and the runs
wins_hist = thinkstats2.Hist(wins, label='Wins')
runs_hist = thinkstats2.Hist(runs, label='Runs Scored')
width = 0.45
thinkplot.PrePlot(2, cols=2) 
thinkplot.Hist(wins_hist, align='right', width=width)
thinkplot.Hist(runs_hist, align='left', width=width)
thinkplot.Config(xlabel='Result', ylabel='Probability')


# In[29]:

# Building the PMFs of both the wins and the runs
wins_pmf = thinkstats2.Pmf(wins, label='Wins')
runs_pmf = thinkstats2.Pmf(runs, label='Runs')

width=0.45
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(wins_pmf, align='right', width=width)
thinkplot.Hist(runs_pmf, align='left', width=width)
thinkplot.Config(xlabel='Result', ylabel='PMF')


# In[30]:

# Building the CDFs of both the wins and the runs
wins_cdf = thinkstats2.Cdf(wins, label='Wins')
runs_cdf = thinkstats2.Cdf(runs, label='Runs')

thinkplot.PrePlot(2)
thinkplot.Cdfs([wins_cdf, runs_cdf])
thinkplot.Config(xlabel='Result', ylabel='CDF')


# In[32]:

(data.W.values) # I will use these numpy arrays later, that's why i created them here.


# In[34]:

(data.R.values)


# In[46]:

wins_sample = data.W.values
runs_sample = data.R.values


# In[50]:

# In this part I will model the distributions. 
cdf_for_wins = thinkstats2.Cdf(wins_sample, label='Wins')
thinkplot.Cdf(cdf_for_wins)
thinkplot.Config(xlabel='Result', ylabel='CDF')


# In[51]:

cdf_for_runs = thinkstats2.Cdf(runs_sample, label='Runs')
thinkplot.Cdf(cdf_for_runs)
thinkplot.Config(xlabel='Result', ylabel='CDF')


# In[52]:

thinkplot.Cdf(cdf_for_wins, complement=True,label='Wins') # CCDF on a log-y scale
thinkplot.Config(xlabel='Result',ylabel='CCDF', yscale='log', loc='upper right')


# In[53]:


thinkplot.Cdf(cdf_for_runs, complement=True,label='Runs')#CCDF on a log-y scale
thinkplot.Config(xlabel='Result', ylabel='CCDF', yscale='log', loc='upper right')


# In[ ]:




# In[47]:

def Cov(xs, ys, meanx=None, meany=None): 
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


# In[65]:

def Jitter(values, jitter=0.5): # the Jitter is plotted using the Jitter module and scatter
    n = len(values)           
    return np.random.normal(0, jitter, n) + values
heights = Jitter(wins, 1.4)
weights = Jitter(runs, 0.5)
thinkplot.Scatter(heights, weights, alpha=0.9, s=40,color='blue')
thinkplot.Config(xlabel='Wins',
                 ylabel='Runs',
                 axis=[0, 210, 20, 200],
                 legend=False)


# In[66]:

# In the Fifth part 
# the covariance is needed in order to compute the correlation
Cov(wins_sample, runs_sample)


# In[67]:

# here we will calculate the correlation 
# this is the code the same from the book
# this is the pearson's correlation 
def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr


# In[68]:


Corr(wins_sample,runs_sample)


# In[69]:

np.corrcoef(wins_sample, runs_sample)


# In[70]:

# this is the other type of correlation which is the spearman correlation
# the code is also exactly the same from the book
def SpearmanCorr(xs, ys):
    xs = pd.Series(xs)
    ys = pd.Series(ys)
    return xs.corr(ys, method='spearman')


# In[71]:


SpearmanCorr(wins_sample, runs_sample)


# In[72]:

# In the last part
class HypothesisTest(object):

    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters=1000):
        self.test_stats = [self.TestStatistic(self.RunModel()) 
                           for _ in range(iters)]

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def TestStatistic(self, data):
        raise UnimplementedMethodException()

    def MakeModel(self):
        pass

    def RunModel(self):
        raise UnimplementedMethodException()


# In[73]:

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data


# In[74]:


Data= wins_sample, runs_sample
HT = DiffMeansPermute(Data)
PValue = HT.PValue()
PValue


# In[75]:


HT.PlotCdf()
thinkplot.Config(xlabel='first test statistic ',ylabel='CDF')


# In[76]:

class DiffMeansOneSided(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat


# In[80]:

Ht = DiffMeansOneSided(Data)
Pvalue = Ht.PValue()
Pvalue
Ht.PlotCdf()
thinkplot.Config(xlabel='second test statistic',ylabel='CDF')


# In[81]:

class DiffStdPermute(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat


# In[82]:

ht = DiffStdPermute(Data)
Pvalue = ht.PValue()
Pvalue
ht.PlotCdf()
thinkplot.Config(xlabel='third test statistic',ylabel='CDF')


# In[ ]:



