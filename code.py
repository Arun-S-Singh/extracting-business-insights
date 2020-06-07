# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visual_summary(type_, df, col):
    """Summarize the Data using Visual Method.
    
    This function accepts the type of visualization, the data frame and the column to be summarized.
    It displays the chart based on the given parameters.
    
    Keyword arguments:
    type_ -- visualization method to be used
    df -- the dataframe
    col -- the column in the dataframe to be summarized
    """
    fig, ax = plt.subplots(figsize=(7, 7)) 

    if type_ == 'bar':
        ax = df.groupby(col)['case'].count().plot(kind=type_)
    elif type_ == 'hist':
        ax = df[col].plot.hist()
    plt.show()


def central_tendency(type_, df, col):
    """Calculate the measure of central tendency.
    
    This function accepts the type of central tendency to be calculated, the data frame and the required column.
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column in the dataframe to do the calculations
    
    Returns:
    cent_tend -- the calculated measure of central tendency
    """
    cent_tend = ''
    if type_ =='mean':
        cent_tend = np.around(df[col].mean(),2)
    elif type_ == 'mode':
        cent_tend = df[col].mode()
    elif type_ == 'median':
        cent_tend = df[col].median()
    
    print('Central tendency {0} for {1} = {2}'.format(type_,col,cent_tend))
    
    return cent_tend

    
    


def measure_of_dispersion(type_, df, col):
    """Calculate the measure of dispersion.
    
    This function accepts the measure of dispersion to be calculated, the data frame and the required column(s).
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated - range, MAD, std dev, CV, iqr and cov 
    df -- the dataframe
    col -- the column(s) in the dataframe to do the calculations, this is a list with 2 elements if we want to calculate covariance
    
    Returns:
    disp -- the calculated measure of dispersion
    """
    mean = ''
    if type_ == 'range':
        disp = df[col].max() - df[col].min()
    elif type_ == 'mad':
        #mean = df[col].sum()/len(df)
        #disp = np.sum(np.absolute([df[col] - mean]))/len(df)
        disp = df[col].mad()
    elif type_ == 'std':
        disp = df[col].std()
    elif type_ == 'iqr':
        disp = df[col].quantile(0.75) - df[col].quantile(0.25)
    elif type_ == 'cv':
        disp = (df[col].std()/df[col].mean())*100
    elif type_ == 'cov':
        disp = df[col[0]].cov(df[col[1]])
    
    disp = np.around(disp,2)

    print('dispersion {0} for {1} = {2}'.format(type_,col,disp))
    
    return disp


def calculate_correlation(type_, df, col1, col2):
    """Calculate the defined correlation coefficient.
    
    This function accepts the type of correlation coefficient to be calculated, the data frame and the two column.
    It returns the calculated coefficient.
    
    Keyword arguments:
    type_ -- type of correlation coefficient to be calculated
    df -- the dataframe
    col1 -- first column
    col2 -- second column
    
    Returns:
    corr -- the calculated correlation coefficient
    """
    corr = df[col1].corr(df[col2],method=type_)
    corr = np.around(corr,2)

    print('({0}) Corelation coefficient of {1} and {2} = {3}'.format(type_,col1,col2,corr))
    
    return corr



def calculate_probability_discrete(data, event):
    """Calculates the probability of an event from a discrete distribution.
    
    This function accepts the distribution of a variable and the event, and returns the probability of the event.
    
    Keyword arguments:
    data -- series that contains the distribution of the discrete variable
    event -- the event for which the probability is to be calculated
    
    Returns:
    prob -- calculated probability fo the event
    """
    prob = data.value_counts()[event]/len(data)
    
    prob = np.around(prob,2)
    print('Probablity of {0} = {1}'.format(event,prob))
    
    return prob






def event_independence_check(prob_event1, prob_event2, prob_event1_event2):
    """Checks if two events are independent.
    
    This function accepts the probability of 2 events and their joint probability.
    And prints if the events are independent or not.
    
    Keyword arguments:
    prob_event1 -- probability of event1
    prob_event2 -- probability of event2
    prob_event1_event2 -- probability of event1 and event2
    """
    print('Probablity {0} and {1} = {2} and joint probability = {3}'.format(prob_event1,prob_event2,prob_event1*prob_event2,prob_event1_event2))

    return (prob_event1_event2 == prob_event1 * prob_event2)
    


def bayes_theorem(df, col1, event1, col2, event2):
    """Calculates the conditional probability using Bayes Theorem.
    
    This function accepts the dataframe, two columns along with two conditions to calculate the probability, P(B|A).
    You can call the calculate_probability_discrete() to find the basic probabilities and then use them to find the conditional probability.
    
    Keyword arguments:
    df -- the dataframe
    col1 -- the first column where the first event is recorded
    event1 -- event to define the first condition
    col2 -- the second column where the second event is recorded
    event2 -- event to define the second condition
    
    Returns:
    prob -- calculated probability for the event1 given event2 has already occured
    """
    num = (len(df[(df[col1]==event1) & (df[col2]==event2)])/len(df))
    deno = (len(df[df[col2]==event2])/len(df))

    prob = num /deno
    prob = np.around(prob,2)
    return prob


# Load the dataset
df = pd.read_csv(path)
#print(df.head())

# Using the visual_summary(), visualize the distribution of the data provided.
# You can also do it at country level or based on years by passing appropriate arguments to the fuction.

visual_summary('bar',df,'country')
visual_summary('hist',df,'year')


# You might also want to see the central tendency of certain variables. Call the central_tendency() to do the same.
# This can also be done at country level or based on years by passing appropriate arguments to the fuction.

central_tendency('mean',df,'exch_usd')
central_tendency('median',df,'exch_usd')
central_tendency('mode',df,'exch_usd')

central_tendency('mode',df,'banking_crisis')
central_tendency('median',df[df['country']=='Algeria'],'exch_usd')


# Measures of dispersion gives a good insight about the distribution of the variable.
# Call the measure_of_dispersion() with desired parameters and see the summary of different variables.

measure_of_dispersion('range',df,'exch_usd')
measure_of_dispersion('mad',df,'exch_usd')
measure_of_dispersion('std',df,'exch_usd')
measure_of_dispersion('iqr',df,'exch_usd')
measure_of_dispersion('cv',df,'exch_usd')
measure_of_dispersion('cov',df,['exch_usd','inflation_annual_cpi'])


# There might exists a correlation between different variables. 
# Call the calculate_correlation() to check the correlation of the variables you desire.

calculate_correlation('pearson',df,'exch_usd','inflation_annual_cpi')
calculate_correlation('spearman',df,'exch_usd','inflation_annual_cpi')


# From the given data, let's check the probability of banking_crisis for different countries.
# Call the calculate_probability_discrete() to check the desired probability.
# Also check which country has the maximum probability of facing the crisis.  
# You can do it by storing the probabilities in a dictionary, with country name as the key. Or you are free to use any other technique.

calculate_probability_discrete(df['banking_crisis'],'crisis')

countries = df['country'].unique()

crises={}
for country in countries: 
    crises[country] = calculate_probability_discrete(df[df['country']==country]['banking_crisis'],'crisis')

max_key = max(crises, key=crises.get)
print('Country with maximum probability of banking crisis = {0}'.format(max_key))



# Next, let us check if banking_crisis is independent of systemic_crisis, currency_crisis & inflation_crisis.
# Calculate the probabilities of these event using calculate_probability_discrete() & joint probabilities as well.
# Then call event_independence_check() with above probabilities to check for independence.

p_syst_crisis = calculate_probability_discrete(df['systemic_crisis'],1)
p_curr_crisis = calculate_probability_discrete(df['currency_crises'],1) 
p_infl_crisis = calculate_probability_discrete(df['inflation_crises'],1)
p_bank_crisis = calculate_probability_discrete(df['banking_crisis'],1)

# Calculate the P(A|B)
p_bank_syst_crisis = (len(df[(df['systemic_crisis']==1) & (df['banking_crisis']=='crisis')])/len(df))/p_syst_crisis
p_bank_curr_crisis = (len(df[(df['currency_crises']==1) & (df['banking_crisis']=='crisis')])/len(df))/p_curr_crisis
p_bank_infl_crisis = (len(df[(df['inflation_crises']==1) & (df['banking_crisis']=='crisis')])/len(df))/p_infl_crisis


p_bank_and_syst_crisis = p_bank_syst_crisis * p_syst_crisis
p_bank_and_curr_crisis = p_bank_curr_crisis * p_curr_crisis
p_bank_and_infl_crisis = p_bank_infl_crisis * p_infl_crisis


event_independence_check(p_bank_crisis,p_syst_crisis,p_bank_and_syst_crisis)
event_independence_check(p_bank_crisis,p_curr_crisis,p_bank_and_curr_crisis)
event_independence_check(p_bank_crisis,p_infl_crisis,p_bank_and_infl_crisis)

# Finally, let us calculate the probability of banking_crisis given that other crises (systemic_crisis, currency_crisis & inflation_crisis one by one) have already occured.
# This can be done by calling the bayes_theorem() you have defined with respective parameters.
prob_ = []
prob_.append(bayes_theorem(df,'banking_crisis','crisis','systemic_crisis',1))
prob_.append(bayes_theorem(df,'banking_crisis','crisis','currency_crises',1))
prob_.append(bayes_theorem(df,'banking_crisis','crisis','inflation_crises',1))

print(prob_)
"""
test = []
test.append(p_bank_and_syst_crisis/p_syst_crisis)
test.append(p_bank_and_curr_crisis/p_curr_crisis)
test.append(p_bank_and_infl_crisis/p_infl_crisis)

print(test)
"""
# Code ends


