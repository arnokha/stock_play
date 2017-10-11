## Commonly used functions in my exercises

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats
import random

category_full_names = {'vbd':'very big drop', 'bd':'big drop', 'md':'medium drop', 'sd':'small drop',
                       'vbg':'very big gain', 'bg':'big gain', 'mg':'medium gain', 'sg':'small gain'}

index2category = {0:'bd', 1:'sd', 2:'sg', 3:'bg'}
category2index = {'bd':0, 'sd':1, 'sg':2, 'bg':3}
movement_category_types = ['bd', 'sd', 'sg', 'bg'] ## 4

def ticker_from_csv(csv_string):
    """ 
    The downloaded files come in the form [ticker].csv. 
    We are just stripping off the csv extension and making the ticker uppercase.
    """
    stock_name = csv_string.rsplit('.', 1)[0] ## Peel off the trailing ".csv"
    stock_name = stock_name.rsplit('/', 1)[1] ## Peel off the leading dir name
    return stock_name.upper()

def get_price_movements(df, period=1):
    raise NameError('Renamed to get_price_movement_percentages')

def get_price_movement_percentages(df, period=1):
    """ Get the movement of a stock that's in a data frame. """
    df = df.sort_index(axis=0) ## We want the dates in ascending order
    movement = np.zeros(int(len(df) / period)) ## int() rounds down
    last_price = -1
    count = 0
    for index, row in df.iterrows():
        if (count % period == 0 and count != 0 ):
            i = int((count / period) - 1)
            movement[i] = 100 * row['close'] / last_price - 100
            last_price = row['close']
        elif (count == 0):
            last_price = row['close']
        count += 1

    return movement

def plot_gaussian(x, x_min=-10, x_max=10, n=10000, fill=False, label=''):
    """
    Expects an np array of movement percentages, 
    plots the gaussian kernel density estimate
    """
    ## Learn the kernel-density estimate from the data
    density = stats.gaussian_kde(x)
    
    ## Evaluate the output on some data points
    xs = np.linspace(x_min, x_max, n)
    y = density.evaluate(xs)
    
    ## Create the plot
    if (label != ''):
        plt.plot(xs, y, label=label)
    else:
        plt.plot(xs, y)
    plt.xlabel('Daily Movement Percentage')
    plt.ylabel('Density')
    
    if (fill):
        plt.fill_between(xs, 0, y)


def plot_gaussian_categorical(x, x_min=-10, x_max=10, n=10000, title='', n_cats=8, n_data=-1):
    """
    Expects an np array of movement percentages, 
    plots the gaussian kernel density estimate
    """
    ## Learn the kernel-density estimate from the data
    density = stats.gaussian_kde(x)

    ## Evaluate the output on some data points
    xs = np.linspace(x_min, x_max, n)
    y = density.evaluate(xs)

    ## Create the plot
    plt.plot(xs, y)
    plt.xlabel('Movement Percentage')
    plt.ylabel('Density')

    ## Get stats
    mu, sigma = np.mean(x), np.std(x)

    ## Plot with conditionals
    if n_cats == 8:
        plt.fill_between(xs, 0, y, where= xs < 0, 
                     facecolor='#eeeedd', interpolate=True) ## Small Drop
        plt.fill_between(xs, 0, y, where= xs < (mu - sigma / 2), 
                     facecolor='yellow', interpolate=True) ## Medium Drop
        plt.fill_between(xs, 0, y, where= xs < (mu - sigma), 
                     facecolor='orange', interpolate=True) ## Big Drop
        plt.fill_between(xs, 0, y, where= xs < (mu - 2*sigma), 
                     facecolor='red', interpolate=True) ## Very big drop
        plt.fill_between(xs, 0, y, where= xs > 0, 
                     facecolor='#ebfaeb', interpolate=True) ## Small Gain
        plt.fill_between(xs, 0, y, where= xs > (mu + sigma/2), 
                     facecolor='#b5fbb6', interpolate=True) ## Gain
        plt.fill_between(xs, 0, y, where= xs > (mu + sigma), 
                     facecolor='#6efa70', interpolate=True) ## Big Gain
        plt.fill_between(xs, 0, y, where= xs > (mu + 2*sigma), 
                     facecolor='green', interpolate=True) ## Very Big Gain
    elif n_cats == 4:
        plt.fill_between(xs, 0, y, where= xs < 0, 
                     facecolor='#c64b4b', interpolate=True) ## Small Drop
        plt.fill_between(xs, 0, y, where= xs < (mu - sigma), 
                     facecolor='red', interpolate=True) ## Big Drop
        plt.fill_between(xs, 0, y, where= xs > 0, 
                     facecolor='#58c959', interpolate=True) ## Small Gain
        plt.fill_between(xs, 0, y, where= xs > (mu + sigma), 
                     facecolor='green', interpolate=True) ## Big Gain
    else:
        raise ValueError('Use n_cats 4 or 8')

    ## Label mu and sigma
    plt.text(x_min + 1, max(y) * 0.8, r'$\mu$ = ' + '{0:.2f}'.format(mu))
    plt.text(x_min + 1, max(y) * 0.9, r'$\sigma$ = ' + '{0:.2f}'.format(sigma))
    if n_data > 0:
        plt.text(x_min + 1, max(y) * 0.7, 'n = ' + str(n_data))

    ## Set title if given
    if (len(title) != 0):
        plt.title(title)

def categorize_movements(movements, n_cats=4):
    """
    Given an array of movements, return an array of categories based on how relatively large the 
    movements are.
    """
    mu, sigma = np.mean(movements), np.std(movements)
    categories = []

    if (n_cats == 8):
        for i in range(len(movements)):
            if (movements[i] <= (mu - 2*sigma)): categories.append('vbd')   ## very big drop
            elif (movements[i] <= (mu - sigma)): categories.append('bd')    ## big drop
            elif (movements[i] <= (mu - sigma/2)): categories.append('md')  ## medium drop
            elif (movements[i] < 0): categories.append('sd')                ## small drop
            elif (movements[i] >= (mu + 2*sigma)): categories.append('vbg') ## very big gain
            elif (movements[i] >= (mu + sigma)): categories.append('bg')    ## big gain
            elif (movements[i] >= (mu + sigma/2)): categories.append('mg')  ## medium gain
            elif (movements[i] >= 0): categories.append('sg')               ## small gain
    elif (n_cats == 4):
        for i in range(len(movements)):
            if (movements[i] <= (mu - sigma)): categories.append('bd')   ## big drop
            elif (movements[i] < 0): categories.append('sd')             ## small drop
            elif (movements[i] >= (mu + sigma)): categories.append('bg') ## big gain
            elif (movements[i] >= 0): categories.append('sg')            ## small gain
    else:
        raise ValueError('Currently only 4 and 8 categories are supported')

    return categories

def count_movement_category(categories, cat_to_count):
    """Given a list of categories, return a count of a specific category"""
    count = 0
    for i in range(len(categories)):
        if categories[i] == cat_to_count: count += 1
    return count

def count_two_day_trends(trends, trend_to_count): raise NameError('Renamed to count_trends')

def count_trends(trends, trend_to_count):
    """Given a list of trends, return a count of a specific trend"""
    count = 0
    for i in range(len(trends)):
        if trends[i] == trend_to_count: count += 1
    return count

def get_two_day_trends(categories): raise NameError('No longer in use; use get_trends() instead')

def get_three_day_trends(categories): raise NameError('No longer in use; use get_trends() instead')

def get_trends(categories, trend_length):
    """
    Given a list of movement categories and length of the trend we are looking for, return a list of 
    trends, which is just a underscore seperated concatenation of categories.
    e.g. we have categories = ['a', 'b', 'c', 'd'], and trend_length 2, 
    we would get ['a_b', 'b_c', 'c_d'].
    If instead, trend_length was 3, we would have ['a_b_c', 'b_c_d'].
    """
    trends = []
    for i in range(len(categories) - trend_length + 1):
        trend_string = categories[i]
        for j in range(trend_length - 1):
            trend_string += '_' + categories[i+j+1]
        trends.append(trend_string)
    return trends

def get_trends_all_stocks(period_length, trend_length, all_category_names, n_cats=4):
    """
    Get an aggregate of trends for all stocks, from a specified period_length 
    (1 would be daily, 7 weekly, etc.), a specified trend_length (2 would be looking for two day trends), 
    and a list all_category_names that contains each possible category name.
    
    We return: 
      all_trends          -- The aggregate list of all trends accross stocks
      all_category_counts -- The aggregate count of each category accross stocks
      all_category_probs  -- The probability of each category accross stocks
    """
    g = glob.glob('stock_data/*.csv')
    
    all_movements = []
    all_movement_categories = []
    all_trends = []
    
    all_category_counts = np.zeros(len(all_category_names), dtype=np.int)
    total_count = 0
    
    for i in range(len(g)):
        df = pd.DataFrame()
        df = df.from_csv(g[i])
        
        movements = get_price_movement_percentages(df, period=period_length)
        movement_categories = categorize_movements(movements, n_cats=n_cats)
        
        all_movements.extend(movements)
        all_movement_categories.extend(movement_categories)
        
        for j in range(len(all_category_names)):
            all_category_counts[j] += count_movement_category(movement_categories, all_category_names[j])

        trends = get_trends(movement_categories, trend_length)
        all_trends.extend(trends)
    
    all_category_probs = np.zeros(len(all_category_names), dtype=np.float)
    total_count = len(all_movement_categories)
    for i in range(len(all_category_names)):
        all_category_probs[i] = (all_category_counts[i] / total_count)

    return (all_trends, all_category_counts, all_category_probs, all_movement_categories)

def get_category_probabilities(movement_categories, n_cats=4):
    if n_cats != 4:
        raise ValueError('Only 4 categories supported at this time')

    bd_count = count_movement_category(movement_categories, 'bd')
    sd_count = count_movement_category(movement_categories, 'sd')
    sg_count = count_movement_category(movement_categories, 'sg')
    bg_count = count_movement_category(movement_categories, 'bg')

    total_cat_count = len(movement_categories)

    p_bd = bd_count / total_cat_count
    p_sd = sd_count / total_cat_count
    p_sg = sg_count / total_cat_count
    p_bg = bg_count / total_cat_count

    category_counts = [bd_count, sd_count, sg_count, bg_count]
    category_probabilities = [p_bd, p_sd, p_sg, p_bg]

    return category_probabilities

def plot_two_day_probability_bar_graph(previous_day, count, two_day_trends, 
                                       cat_probs, n_cats=8, show_baseline=True):
    """
    Plot regular probabilities and probabilities conditioned on one event 
    (the previous_day arg)
    """
    two_day_probs = []
    if (n_cats == 8):
        all_categories = ['vbd', 'bd', 'md', 'sd', 'sg', 'mg', 'bg', 'vbg']
    elif(n_cats == 4):
        all_categories = ['bd', 'sd', 'sg', 'bg']
    for next_day in all_categories:
        two_day_name = previous_day +'_' + next_day
        two_day_count = count_trends(two_day_trends, two_day_name)
        two_day_prob = two_day_count / count
        two_day_probs.append(two_day_prob)

    plt.figure(figsize=(11,4))
    if (n_cats == 8):
        categories = ('Very Big Drop', 'Big Drop', 'Medium Drop', 'Small Drop', 
                      'Small Gain', 'Medium Gain', 'Big Gain', 'Very Big Gain')
        ind = np.arange(8)
    elif (n_cats == 4):
        categories = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
        ind = np.arange(4)
    width = 0.25

    if (show_baseline):
        orig_pl = plt.bar(ind+width, cat_probs, width, color='b', label='Original')
    conditioned_pl = plt.bar(ind, two_day_probs, width, color='r', 
                             label='After a ' + category_full_names[previous_day])

    plt.text(0.5, max(two_day_probs) * .95, 'n = ' + '{0:d}'.format(count), 
             ha='center', va='center', weight='medium')
    plt.ylabel('Probabilities')
    plt.title('Probabilities of each Category')
    plt.xticks(ind+width, categories)
    plt.legend()
    #plt.show()

def plot_three_day_probability_bar_graph(previous_day, two_day_trends, three_day_trends, 
                                         movement_categories):
    """
    Plot all of the following together on one figure: regular probabilities, 
    probabilities conditioned on one event (the previous_day arg), and 
    probabilities conditioned on two events (the previous_day, and all possible days before the 
    previous_day)
    """
    import matplotlib.patches as mpatches
    two_day_probs = []
    three_day_probs = []
    all_categories = ['bd', 'sd', 'sg', 'bg']
    cat_count = count_movement_category(movement_categories, previous_day)
    category_probabilities = get_category_probabilities(movement_categories)

    ## Get probabilities after 'previous_day'
    for next_day in all_categories:
        two_day_name = previous_day +'_' + next_day
        two_day_count = count_trends(two_day_trends, two_day_name)
        two_day_prob = two_day_count / cat_count
        two_day_probs.append(two_day_prob)
    
    ## Get probabilities after 'previous_day' and the day before
    for next_day in all_categories:
        for day_before_last in all_categories:  
            three_day_name = day_before_last +'_' + previous_day +'_' + next_day
            three_day_count = count_trends(three_day_trends, three_day_name)

            two_day_name = day_before_last +'_' + previous_day
            three_day_total = 0
            for category in all_categories:
                three_day_total += count_trends(three_day_trends, two_day_name + '_' + category)

            three_day_prob = three_day_count / three_day_total
            three_day_probs.append(three_day_prob)

    fig = plt.figure(figsize=(11,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.9])
    categories = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
    ind = np.arange(4)
    width = 0.1

    ## Plot three day probabilities
    for i in range(int(len(three_day_probs) / 4)):
        pl = ax.bar(ind[i] + 1 * width, three_day_probs[i * 4], width, color='red')
        pl = ax.bar(ind[i] + 2 * width, three_day_probs[i * 4 + 1], width, color='#c64b4b')
        pl = ax.bar(ind[i] + 3 * width, three_day_probs[i * 4 + 2], width, color='#58c959')
        pl = ax.bar(ind[i] + 4 * width, three_day_probs[i * 4 + 3], width, color='green')

    ## Plot two day probability 
    conditioned_pl = ax.bar(ind + (5 * width), two_day_probs, width * 1.5, color='blue')
    orig_pl = ax.bar(ind + (6.5 * width), category_probabilities, width * 1.5, color='black')

    labels = ['Original',
              'After a ' + category_full_names[previous_day], 
              'After a big drop, then a ' + category_full_names[previous_day],
              'After a small drop, then a ' + category_full_names[previous_day], 
              'After a small gain, then a ' + category_full_names[previous_day], 
              'After a big gain, then a ' + category_full_names[previous_day]]
    original_patch = mpatches.Patch(color='black')
    two_day_patch = mpatches.Patch(color='blue')
    bd_x_patch = mpatches.Patch(color='red')
    sd_x_patch = mpatches.Patch(color='#c64b4b')
    sg_x_patch = mpatches.Patch(color='#58c959')
    bg_x_patch = mpatches.Patch(color='green')
    fig.legend([original_patch, two_day_patch, bd_x_patch, sd_x_patch, sg_x_patch, bg_x_patch], 
               labels, 'upper right')

    plt.ylabel('Probabilities')
    plt.title('Probabilities of each Category')
    plt.xticks(ind + 4.5 * width, categories)

#######################
## Practice 10
#######################
def run_random_walks(starting_value, stride_length, p, n_steps, n_trials):
    """
    Run 1D random walks with the following parameters:
     starting_value - Value on the number line at which to start the random walk
     stride_length  - Size of our steps in either direction
     p              - Probability of success
     n_trials       - Number of trials to run
     n_steps        - Number of steps to take on our random walk
      
    NOTE: 0 will be an absorbing state. 
          (Meaning that if we hit 0 we're stuck there)

    Returns the trial_results, which contains the results of each random walk
    """
    trial_results = []

    for i in range(n_trials):
        values = []
        value = starting_value
        for j in range(n_steps):
            values.append(value)
            if (value <= 0):
                value = 0
            elif (random.random() < p):
                value += stride_length
            else:
                value -= stride_length
        trial_results.append(values)

    return trial_results

def run_random_walks_kelly(starting_value, p, n_steps, n_trials):
    """
    Run 1D random walks with the following parameters:
     starting_value - Value on the number line at which to start the random walk
     stride_length  - Size of our steps in either direction
     p              - Probability of success
     n_trials       - Number of trials to run
     n_steps        - Number of steps to take on our random walk
      
    NOTE: 0 will be an absorbing state. 
          (Meaning that if we hit 0 we're stuck there)
    
    Returns the trial_results, which contains the results of each random walk
    """
    trial_results = []

    for i in range(n_trials):
        values = []
        value = starting_value
        for j in range(n_steps):
            values.append(value)
            stride_length = int((2 * p - 1) * value) ## Kelly
            if (value <= 0):
                value = 0
            elif (random.random() < p):
                value += stride_length
            else:
                value -= stride_length
        trial_results.append(values)

    return trial_results

def run_gaussian_random_walks(starting_value, mu, sigma, n_steps, n_trials):
    """
    Run 1D random gaussian walks with the following parameters:
     starting_value - Value on the number line at which to start the random walk
     mu             - Average percent value of stride length
     sigma          - Average percent standard deviation
     n_trials       - Number of trials to run
     n_steps        - Number of steps to take on our random walk
      
    NOTE: 0 will be an absorbing state. 
          (Meaning that if we hit 0 we're stuck there)
    
    Returns the trial_results, which contains the results of each random walk
    We are assuming no "edge" to tilt things in our favor
    """
    trial_results = []

    for i in range(n_trials):
        step_multipliers = np.random.normal(mu, sigma, n_steps) / 100
        values = []
        value = starting_value
        for j in range(n_steps):
            values.append(value)
            if (value <= 0):
                value = 0
            value = value + step_multipliers[j] * value
        trial_results.append(values)

    step_multipliers
    return trial_results

def categorize_movement(movement, mu, sigma, n_cats=4):
    if not (n_cats == 4):
        raise ValueError('Only 4 categories supported at this time')

    if (movement <= (mu - sigma)):
        category = 'bd'  ## big drop
    elif (movement <= mu):
        category = 'sd'  ## small drop
    elif (movement >= (mu + sigma)):
        category = 'bg'  ## big gain
    elif (movement >= mu):
        category = 'sg'  ## small gain

    return category

def choose_category(labels, probabilities):
    num = np.random.rand(1)[0]
    for i in range(len(probabilities)):
        num = num - probabilities[i]
        if num <= 0:
            return labels[i]
        
    ## Probabilities didn't sum perfectly to one
    return np.random.choice(labels, 1)[0]

def generate_next_two_day_step(previous_step, two_day_probs, mu, sigma):
    conditional_probabilities = {'bd':two_day_probs[0:4], 
                                 'sd':two_day_probs[4:8],
                                 'sg':two_day_probs[8:12],
                                 'bg':two_day_probs[12:16]}
    conditional_probability = conditional_probabilities[categorize_movement(previous_step, mu, sigma)]
    
    choice = choose_category(['bd', 'sd', 'sg', 'bg'], conditional_probability)
    
    random_samples = np.random.normal(mu, sigma, 1000)
    
    ## Draw on random samples until we get a result of the correct category
    for i in range(len(random_samples)):
        if (categorize_movement(random_samples[i], mu, sigma) == choice):
            #print(choice)
            #print(random_samples[i])
            return random_samples[i]
        
    ## Very unlikely to happen, but will catch in the case none of the samples 
    ## have the category we're looking for
    return 0

def get_probabilities(two_day_trends, categories, n_cats=4):
    two_day_probs = []
    if (n_categories == 4):
        all_categories = ['bd', 'sd', 'sg', 'bg']
    else:
        raise ValueError('Only four categories are supported at this time')
        
    for first_day in all_categories:
        first_day_count = count_movement_category(categories, first_day)
        for next_day in all_categories:
            two_day_name = first_day +'_' + next_day
            two_day_count = count_trends(two_day_trends, two_day_name)
            two_day_prob = two_day_count / first_day_count
            two_day_probs.append(two_day_prob)

    return two_day_probs

def run_two_day_momentum_simulation(prior_daily_movements, starting_value, mu, sigma, n_steps, n_trials):
    ## Get categories and trends
    prior_movement_categories = categorize_movements(prior_daily_movements, n_cats=4)
    prior_two_day_trends = get_two_day_trends(prior_movement_categories)
    two_day_probs = get_probabilities(prior_two_day_trends, prior_movement_categories)
    
    trials = []
    for i in range(n_trials):
        first_step = generate_next_two_day_step(prior_daily_movements[-1], two_day_probs, mu, sigma)
        #print(first_step)
        steps = [first_step]

        for i in range(n_steps - 1):
            steps.append(generate_next_two_day_step(steps[i], two_day_probs, mu, sigma))
        
        trials.append(simulate_movements(steps, starting_value))

    return trials

def get_three_day_probabilities(three_day_trends, two_day_name, categories, n_cats=4):
    """
    Returns the probability distribution for the third day given the previous 
    two
    """
    two_day_probs = []
    if (n_categories == 4):
        all_categories = ['bd', 'sd', 'sg', 'bg']
    else:
        raise ValueError('Only four categories are supported at this time')
        
    three_day_counts = []
    total = 0
    for next_day in all_categories:
        three_day_name = two_day_name + '_' + next_day
        three_day_count = count_trends(three_day_trends, three_day_name)
        total += three_day_count
        three_day_counts.append(three_day_count)

    three_day_probs = []
    [three_day_probs.append(three_day_counts[i] / total) for i in range(len(three_day_counts))]
    return three_day_probs

def generate_next_three_day_step(step_before_last, previous_step, three_day_probability, mu, sigma):
    two_day_name = categorize_movement(step_before_last, mu, sigma) + '_' + \
                   categorize_movement(previous_step, mu, sigma)
    choice = choose_category(['bd', 'sd', 'sg', 'bg'], three_day_probability)
    random_samples = np.random.normal(mu, sigma, 1000)
    
    ## Draw on random samples until we get a result of the correct category
    for i in range(len(random_samples)):
        if (categorize_movement(random_samples[i], mu, sigma) == choice):
            return random_samples[i]
        
    ## Very unlikely to happen, but will catch in the case none of the samples 
    ## have the category we're looking for
    return 0

def run_three_day_momentum_simulation(prior_daily_movements, starting_value, 
                                      mu, sigma, n_steps, n_trials):
    ## Get categories and trends
    prior_movement_categories = categorize_movements(prior_daily_movements)
    prior_three_day_trends = get_three_day_trends(prior_movement_categories)
    
    trials = []
    
    ## Collect a dictionay of three day probabilities
    all_categories = ['bd', 'sd', 'sg', 'bg']
    three_day_probs = {}
    for first_day in all_categories:
        for next_day in all_categories:
            two_day_name = first_day + '_' + next_day
            three_day_probs[two_day_name] = get_three_day_probabilities(prior_three_day_trends, 
                                                                        two_day_name, 
                                                                        prior_movement_categories)  
    ## Generate steps based on the movements of the prior two days
    for i in range(n_trials):
        two_day_name = categorize_movement(prior_daily_movements[-2], mu, sigma) + '_' + \
                       categorize_movement(prior_daily_movements[-1], mu, sigma)
        three_day_prob = three_day_probs[two_day_name]
        first_step = generate_next_three_day_step(prior_daily_movements[-2], prior_daily_movements[-1], 
                                                  three_day_prob, mu, sigma)
        
        two_day_name = categorize_movement(prior_daily_movements[-2], mu, sigma) + '_' + \
                       categorize_movement(prior_daily_movements[-1], mu, sigma)
        three_day_prob = three_day_probs[two_day_name]
        second_step = generate_next_three_day_step(prior_daily_movements[-1], first_step, three_day_prob, 
                                                   mu, sigma)
        
        steps = [first_step, second_step]

        for i in range(n_steps - 2):
            two_day_name = categorize_movement(steps[i], mu, sigma) + '_' + \
                           categorize_movement(steps[i+1], mu, sigma)
            three_day_prob = three_day_probs[two_day_name]
            steps.append(generate_next_three_day_step(steps[i], steps[i+1], three_day_prob, mu, sigma))
        
        trials.append(simulate_movements(steps, starting_value))

    return trials

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Intra-day Range functions (Nb 11)
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def get_intra_day_range(df):
    df = df.sort_index(axis=0)
    intra_day_range = np.zeros(len(df))
    i = 0
    for index, row in df.iterrows():
        intra_day_range[i] = row['high'] - row['low']
        i += 1
    
    return intra_day_range

def get_intra_day_range_percentage(df):
    df = df.sort_index(axis=0)
    intra_day_range = np.zeros(len(df))
    i = 0
    for index, row in df.iterrows():
        intra_day_range[i] = 100 * (row['high'] - row['low']) / ((row['high'] + row['low']) / 2)
        i += 1
    
    return intra_day_range

def categorize_ranges(ranges):
    """
    Given an array of ranges, return an array of categories based on how 
    relatively large the ranges are
    """
    mu, sigma = np.mean(ranges), np.std(ranges)
    categories = []
    
    for i in range(len(ranges)):
        if (ranges[i] <= (mu - sigma)):
            categories.append('vs') ## very small
        elif (ranges[i] < mu):
            categories.append('s')  ## small drop
        elif (ranges[i] >= (mu + sigma)):
            categories.append('vl') ## very large
        elif (ranges[i] >= mu):
            categories.append('l')  ## small gain
        else:
            print("didn't fit")
    
    return categories

def count_range_category(categories, cat_to_count):
    count = 0
    for i in range(len(categories)):
        if categories[i] == cat_to_count:
            count = count + 1
    return count

def count_trends(trends, trend_to_count):
    count = 0
    for i in range(len(trends)):
        if trends[i] == trend_to_count:
            count = count + 1
    return count

def get_two_day_range_trends(range_categories, movement_categories):
    two_day_trends = []
    for i in range(len(range_categories) - 1):
        two_day_trends.append(str(range_categories[i]) + '_' + str(movement_categories[i+1]))
    return two_day_trends

def plot_probability_bar_graph_ranges(name, count, two_day_trends, show_baseline=True, n_cats=4):
    if (n_cats != 4):
        raise ValueError('Only four categories are supported at this time')
        
    two_day_probs = []
    range_full_names = {'vs':'very small', 's':'small', 'l':'large', 'vl':'very large'}
    
    all_categories = ['bd', 'sd', 'sg', 'bg']
    for next_day in all_categories:
        two_day_name = name +'_' + next_day
        two_day_count = count_trends(two_day_trends, two_day_name)
        two_day_prob = two_day_count / count
        two_day_probs.append(two_day_prob)

    plt.figure(figsize=(11,4))
    movement_category_names = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
    range_category_names = ('Very Small', 'Small', 'Large', 'Very Large')
    ind = np.arange(4)
    width = 0.25
    if (show_baseline):
        orig_pl = plt.bar(ind+width, movement_cat_probs, width, color='b', label='Original')
    conditioned_pl = plt.bar(ind, two_day_probs, width, color='r', 
                             label='After a ' + range_full_names[name] + ' ID range')
    plt.ylabel('Probabilities')
    plt.title('Probabilities of each Category')
    plt.xticks(ind+width, movement_category_names)
    plt.legend()
    plt.show()


def get_idr_trends_all_stocks(period_length, all_category_names, trend_length=2, n_cats=4):
    """
    Get an aggregate of trends for all stocks, from a specified period_length 
    (1 would be daily, 7 weekly, etc.),
    a specified trend_length(2 would be looking for two day trends), and a 
    list all_category_names that contains each possible category name.
    
    We return: 
      all_trends          -- The aggregate list of all trends accross stocks
      all_category_counts -- The aggregate count of each category accross stocks
      all_category_probs  -- The probability of each category accross stocks
    """
    if (trend_length != 2):
        raise ValueError('Trend length must be two for now')
    if (n_cats != 4):
        raise ValueError('Number of categories musr be four for now')
    
    g = glob.glob('stock_data/*.csv')
    
    all_range_categories = []
    all_trends = []
    
    all_range_category_counts = np.zeros(len(all_category_names), dtype=np.int)
    total_count = 0
    
    for i in range(len(g)):
        df = pd.DataFrame()
        df = df.from_csv(g[i])
        
        movements = get_price_movement_percentages(df, period=period_length)
        movement_categories = categorize_movements(movements, n_cats=n_cats)
        
        range_categories = categorize_ranges(get_intra_day_range_percentage(df))
        #
        #all_movements.extend(movements)
        all_range_categories.extend(movement_categories)
        
        for j in range(len(all_category_names)):
            #print(all_category_names[j])
            all_range_category_counts[j] += count_range_category(range_categories, all_category_names[j])
        #print(all_range_category_counts)
        trends = get_two_day_range_trends(range_categories, movement_categories)
        all_trends.extend(trends)
    
    all_category_probs = np.zeros(len(all_category_names), dtype=np.float)
    total_count = len(all_range_categories)
    for i in range(len(all_category_names)):
        all_category_probs[i] = (all_range_category_counts[i] / total_count)

    return (all_trends, all_range_category_counts, all_category_probs, all_range_categories)

##-=-=-=-=-=-=-=-=-=-=-=-=-=
## Volume functions (Nb 12)
##-=-=-=-=-=-=-=-=-=-=-=-=-=
def get_volume(df):
    df = df.sort_index(axis=0) ## We want dates in sequential order
    return df['volume'].as_matrix()

def get_relative_volume(df, relative_period=50):
    df = df.sort_index(axis=0) ## We want dates in sequential order
    absolute_volumes = df['volume'].as_matrix() #.astype(float)
    relative_volumes = np.zeros(len(absolute_volumes))

    ## For the middle, branch out this far in either direction
    half = int(relative_period / 2) 
    
    for i in range(len(absolute_volumes)):
        end_relative_period = len(absolute_volumes) - relative_period
        ## Middle
        if i >= relative_period and i < end_relative_period:
            #print('middle')
            count = 0
            total_volume = 0
            for j in range(half):
                total_volume += absolute_volumes[i-j]
                count += 1
            for j in range(half):
                total_volume += absolute_volumes[i+j+1]
                count += 1
            avg_volume = total_volume / count
            relative_volumes[i] = absolute_volumes[i] / avg_volume
        ## Beginning
        elif i < relative_period:
            #print('beginning')
            count = 0
            total_volume = 0
            for j in range(relative_period):
                total_volume += absolute_volumes[i+j]
                count += 1
            avg_volume = total_volume / count
            relative_volumes[i] = absolute_volumes[i] / avg_volume
        ## End
        elif i >= (len(absolute_volumes) - relative_period):
            #print('end')
            count = 0
            total_volume = 0
            for j in range(relative_period):
                total_volume += absolute_volumes[i-j]
                count += 1
            avg_volume = total_volume / count
            relative_volumes[i] = absolute_volumes[i] / avg_volume
        else:
            print('something went wrong')
        
    return relative_volumes


def categorize_volumes(ranges):
    """
    Given an array of ranges, return an array of categories based on how 
    relatively large the ranges are
    """
    mu, sigma = np.mean(ranges), np.std(ranges)
    categories = []
    
    for i in range(len(ranges)):
        if (ranges[i] <= (mu - sigma)):
            categories.append('vl')  ## very low
        elif (ranges[i] < mu):
            categories.append('l')   ## low
        elif (ranges[i] >= (mu + sigma)):
            categories.append('h')   ## high
        elif (ranges[i] >= mu):
            categories.append('vh')  ## very high
        else:
            print("didn't fit")
    
    return categories

def count_volume_category(categories, cat_to_count):
    count = 0
    for i in range(len(categories)):
        if categories[i] == cat_to_count:
            count += 1
    return count

def get_two_day_volume_trends(vol_categories, movement_categories):
    two_day_trends = []
    for i in range(len(vol_categories) - 1):
        two_day_trends.append(vol_categories[i] + '_' + movement_categories[i+1])
    return two_day_trends

def plot_probability_bar_graph_volumes(name, count, two_day_trends, show_baseline=True, n_cats=4):
    if (n_cats != 4):
        raise ValueError('Only four categories are supported at this time')
        
    two_day_probs = []
    volume_full_names = {'vl':'very low', 'l':'low', 'h':'high', 'vh':'very high'}
    
    all_categories = ['bd', 'sd', 'sg', 'bg']
    for next_day in all_categories:
        two_day_name = name +'_' + next_day
        two_day_count = count_trends(two_day_trends, two_day_name)
        two_day_prob = two_day_count / count
        two_day_probs.append(two_day_prob)

    plt.figure(figsize=(11,4))
    movement_category_names = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
    volume_category_names = ('Very Low', 'Low', 'High', 'Very High')
    ind = np.arange(4)
    width = 0.25
    if (show_baseline):
        orig_pl = plt.bar(ind+width, movement_cat_probs, width, color='b', label='Original')
    conditioned_pl = plt.bar(ind, two_day_probs, width, color='r', 
                             label='After a ' + volume_full_names[name] + ' volume day')
    plt.ylabel('Probabilities')
    plt.title('Probabilities of each Category')
    plt.xticks(ind+width, movement_category_names)
    plt.legend()
    plt.show()

def get_volume_trends_all_stocks(period_length, all_category_names, 
                                 trend_length=2, n_cats=4, relative_period=50):
    """
    Get an aggregate of trends for all stocks, from a specified period_length 
    (1 would be daily, 7 weekly, etc.),
    a specified trend_length(2 would be looking for two day trends), 
    and a list all_category_names that contains each possible category name.
    
    We return: 
      all_trends          -- The aggregate list of all trends accross stocks
      all_category_counts -- The aggregate count of each category accross stocks
      all_category_probs  -- The probability of each category accross stocks
    """
    if (trend_length != 2):
        raise ValueError('Trend length must be two for now')
    if (n_cats != 4):
        raise ValueError('Number of categories musr be four for now')
    
    g = glob.glob('stock_data/*.csv')
    
    all_volume_categories = []
    all_trends = []
    
    all_volume_category_counts = np.zeros(len(all_category_names), dtype=np.int)
    total_count = 0
    
    for i in range(len(g)):
        df = pd.DataFrame()
        df = df.from_csv(g[i])
        
        movements = get_price_movement_percentages(df, period=period_length)
        movement_categories = categorize_movements(movements, n_cats=n_cats)
        volume_categories = categorize_volumes(get_relative_volume(df, relative_period=relative_period))
        all_volume_categories.extend(movement_categories)
        
        for j in range(len(all_category_names)):
            all_volume_category_counts[j] += count_volume_category(volume_categories, 
                                                                   all_category_names[j])
        trends = get_two_day_volume_trends(volume_categories, 
                                           movement_categories)
        all_trends.extend(trends)
    
    all_category_probs = np.zeros(len(all_category_names), dtype=np.float)
    total_count = len(all_volume_categories)
    for i in range(len(all_category_names)):
        all_category_probs[i] = (all_volume_category_counts[i] / total_count)

    return (all_trends, all_volume_category_counts, all_category_probs, all_volume_categories)

##-=-=-=-=-=-=-=-=-=-=-=-=-=
##  (Nb 13)
##-=-=-=-=-=-=-=-=-=-=-=-=-=
def get_single_day_probabilities(movement_categories):
    movement_category_types = ['bd', 'sd', 'sg', 'bg']
    single_day_counts = []
    single_day_probabilities = []
    total = 0
    
    for cat in movement_category_types:
        count = count_movement_category(movement_categories, cat)
        single_day_counts.append(count)
        total += count
    
    for count in single_day_counts:
        single_day_probabilities.append(count/total)
        
    return single_day_probabilities

def get_probabilities_after_event(previous_event_category, trends, movement_categories):
    """
    Given an event that occured the previous day, return the probabilities of 
    the next day's movement categories conditioned on said event.
    
    Arguments:
      previous_event_category - The category of the event we observed the previous day 
                                (or two days in the case of three day momentum)
      trends - All two (or three) day trends that were observed for this event type.
      movement_categories - all daily movement categories that were observed
    
    Returns:
      next_day_movement_probabilities - Probabilities of each of the next day's categories conditioned on 
                                        the previous event category
    """
    movement_category_types = ['bd', 'sd', 'sg', 'bg']
    next_day_movement_probabilities = []

    trend_total = 0
    for category in movement_category_types:
        trend_total += count_trends(trends, previous_event_category + '_' + category)
    
    for next_day in movement_category_types:
        trend_name = previous_event_category + '_' + next_day
        #print(trend_name)
        trend_count = count_trends(trends, trend_name)
        #print(trend_count)
            
        trend_prob = trend_count / trend_total
        next_day_movement_probabilities.append(trend_prob)
        
    return next_day_movement_probabilities

def select_data_sample(data, sample_size, data2=None):
    ## We are going to omit the last sample_size elements, 
    ## so that  if we start the sample towards the end we won't run out of 
    ## elements
    sub_sample = data[0:-sample_size]
    random_index = random.choice(list(enumerate(sub_sample)))[0]
    if data2 is None:
        return data[random_index:random_index+sample_size]
    else:
        return data[random_index:random_index+sample_size], data2[random_index:random_index+sample_size]

def get_next_day_probability(probabilities_given_by_model, previous_days):
    if len(previous_days) == 2:
        index = category2index[previous_days[0]] * 4 + category2index[previous_days[1]]
    elif len(previous_days) == 1:
        index = category2index[previous_days[0]]
    elif len(previous_days) == 0:
        return probabilities_given_by_model
    else:
        raise ValueError('So far, only one to three day models are ' + \
               'supported. Please set previous_days to a list of length 0 to 2')
    
    return probabilities_given_by_model[index]

def build_model_probabilities(movement_categories, trends, n_day_model, 
                              previous_category_types=['bd', 'sd', 'sg', 'bg']):
    
    ## Three day model
    if n_day_model == 3:
        three_day_probs = []
        for cat in previous_category_types:
            for cat2 in previous_category_types:
                two_day_name = cat + '_' + cat2
                three_day_probs.append(get_probabilities_after_event(two_day_name, trends, 
                                                                     movement_categories))
        return three_day_probs
    
    ## Two day model
    elif n_day_model == 2:
        two_day_probs = []
        for cat in previous_category_types:
            two_day_probs.append(get_probabilities_after_event(cat, trends, movement_categories))
        return two_day_probs
    
    ## One day model
    elif n_day_model == 1:
        one_day_probs = get_single_day_probabilities(movement_categories)
        return one_day_probs

    else:
        raise ValueError('So far, only one to three day models are ' + 
                         'supported. Please set n_day_model between 1 and 3')
        
    return

def random_sample_tests_m1_m2(movement_categories, m1_probs, m1_n_day_model, m2_probs, m2_n_day_model, 
                              sample_size=50, n_tries=10000):
    m1_wins = 0
    m2_wins = 0
    n_draws = 0
    
    for a in range(n_tries):
        sample = select_data_sample(movement_categories, sample_size)
        sample_data_probabilities = get_single_day_probabilities(sample)
        
        m1_round_score = 0
        m2_round_score = 0
        
        ## This is so models have enough data to "look back" and predict the 
        ##following day
        n_lookback_days = max(m1_n_day_model, m2_n_day_model) - 1
        round_length = len(sample) - n_lookback_days 
        
        for i in range(round_length):
            if (n_lookback_days == 1):
                prev_day = sample[i]
                next_day = sample[i+1]
            elif (n_lookback_days == 2):
                day_before_last = sample[i]
                prev_day = sample[i+1]
                next_day = sample[i+2]
            else:
                raise ValueError('This function was meant to test one, ' + 
                                'two, and three day models against each other.')
                
            if (m1_n_day_model == 1):
                m1_next_day_probs = get_next_day_probability(m1_probs, [])
            elif (m1_n_day_model == 2):
                m1_next_day_probs = get_next_day_probability(m1_probs, 
                                                             [prev_day])
            elif (m1_n_day_model == 3):
                m1_next_day_probs = get_next_day_probability(m1_probs, 
                                            [day_before_last, prev_day])
            if (m2_n_day_model == 1):
                m2_next_day_probs = get_next_day_probability(m2_probs, [])
            elif (m2_n_day_model == 2):
                m2_next_day_probs = get_next_day_probability(m2_probs, 
                                                             [prev_day])
            elif (m2_n_day_model == 3):
                m2_next_day_probs = get_next_day_probability(m2_probs, 
                                                    [day_before_last, prev_day])
        
            ## Weight correct answers on larger movements more heavily
            ## In the case of a tie, don't award any points
            # M1 wins
            if m1_next_day_probs[category2index[next_day]] > \
               m2_next_day_probs[category2index[next_day]]:
                if(next_day == 'bg' or next_day == 'bd'):
                    m1_round_score += 2
                else:
                    m1_round_score += 1
            # M2 wins
            elif m2_next_day_probs[category2index[next_day]] > \
                 m1_next_day_probs[category2index[next_day]]:
                if(next_day == 'bg' or next_day == 'bd'):
                    m2_round_score += 2
                else:
                    m2_round_score += 1

        if m1_round_score > m2_round_score:
            m1_wins += 1
        elif m2_round_score > m1_round_score:
            m2_wins += 1
        else:
            n_draws += 1
    
    return m1_wins, m2_wins, n_draws


##
## Validation testing
##
def get_train_valid_trends_all_stocks(period_length, trend_length, all_category_names, 
                                      training_split_percentage=0.80, n_cats=4):
    """
    Get an aggregate of trends for all stocks, from a specified period_length 
    (1 would be daily, 7 weekly, etc.), a specified trend_length (2 would be looking for two day trends), 
    and a list all_category_names that contains each possible category name.
    
    We return: 
      all_trends          -- The aggregate list of all trends accross stocks
      all_category_counts -- The aggregate count of each category accross stocks
      all_category_probs  -- The probability of each category accross stocks
    """
    g = glob.glob('stock_data/*.csv')
    
    train_all_movement_categories = []
    train_all_trends = []
    valid_all_movement_categories = []
    valid_all_trends = []
    
    train_all_category_counts = np.zeros(len(all_category_names), dtype=np.int)
    valid_all_category_counts = np.zeros(len(all_category_names), dtype=np.int)
    #train_total_count = 0
    #valid_total_count = 0

    #train_split = int(training_split_percentage / 100) * len(movements)
    #valid_split = int((100 - training_split_percentage) / 10)
    #print(range(len(g)))

    for i in range(len(g)):
        df = pd.DataFrame()
        df = df.from_csv(g[i])
        
        movements = get_price_movement_percentages(df, period=period_length)
        #movement_categories = categorize_movements(movements, n_cats=n_cats)
        train_split = int(training_split_percentage * len(movements))
        train_movement_categories = categorize_movements(movements[0:train_split])
        valid_movement_categories = categorize_movements(movements[train_split+1:len(movements)])
        
        train_all_movement_categories.extend(train_movement_categories)
        valid_all_movement_categories.extend(valid_movement_categories)
        
        for j in range(len(all_category_names)):
            train_all_category_counts[j] += \
              count_movement_category(train_movement_categories, all_category_names[j])
            valid_all_category_counts[j] += \
              count_movement_category(valid_movement_categories, all_category_names[j])

        train_trends = get_trends(train_movement_categories, trend_length)
        valid_trends = get_trends(valid_movement_categories, trend_length)
        train_all_trends.extend(train_trends)
        valid_all_trends.extend(valid_trends)
    
    train_all_category_probs = np.zeros(len(all_category_names), dtype=np.float)
    valid_all_category_probs = np.zeros(len(all_category_names), dtype=np.float)
    train_total_count = len(train_all_movement_categories)
    valid_total_count = len(valid_all_movement_categories)
    for i in range(len(all_category_names)):
        train_all_category_probs[i] = (train_all_category_counts[i] / train_total_count)
        valid_all_category_probs[i] = (valid_all_category_counts[i] / valid_total_count)

    return (train_all_trends, train_all_category_counts, 
            train_all_category_probs, train_all_movement_categories,
            valid_all_trends, valid_all_category_counts, 
            valid_all_category_probs, valid_all_movement_categories)

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
## Nbs 05x2 and 08x2: Linear predictions rather than categorical
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
def get_trends_linear(movement_categories, movement_percentages, trend_length):
    """
    Given a list of movement categories, a list of movement percentages, and
    the length of the trend we are looking for, return a list of trend_length-1
    tuples containing trend categories and then the following days movement percentage.

    e.g. We have categories = ['a', 'b', 'a', 'c', 'a'], and associated movement percentages 
         [-1, 1, -2, 5, -1]. If the trend_length we are looking at is 2, we would get
         [('a',1), ('b',-2), ('a',5), ('c',-1)] 

         If instead, trend_length was 3, we would have [('a_b', -2), ('b_a', 5), ('a_c', -1)].
    """
    trends = []
    for i in range(len(movement_categories) - trend_length + 1):
        trend_string = movement_categories[i]
        counter = 1
        for _ in range(trend_length - 2):
            trend_string += '_' + movement_categories[i+counter]
            counter += 1
        trend_and_movement = (trend_string, movement_percentages[i+counter])
        trends.append(trend_and_movement)
    return trends

def get_movements_after_trend(trend, trends_and_movements):
    """Get all stock movement percentages after the given trend is observed"""
    movements_after_trend = []
    for i in range(len(trends_and_movements)):
        if trend == trends_and_movements[i][0]:
            movements_after_trend.append(trends_and_movements[i][1])
            
    return movements_after_trend

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
## Nbs 16: Model classes. TODO Clean them up
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
def p_array_to_dict(p, trend_length):
    p_dict = {}
    if (trend_length == 1):
        p_dict['bd'] = p[0]
        p_dict['sd'] = p[1]
        p_dict['sg'] = p[2]
        p_dict['bg'] = p[3]

    elif (trend_length == 2):
        p_dict['bd_bd'] = p[0][0]
        p_dict['bd_sd'] = p[0][1]
        p_dict['bd_sg'] = p[0][2]
        p_dict['bd_bg'] = p[0][3]
        p_dict['sd_bd'] = p[1][0]
        p_dict['sd_sd'] = p[1][1]
        p_dict['sd_sg'] = p[1][2]
        p_dict['sd_bg'] = p[1][3]
        p_dict['sg_bd'] = p[2][0]
        p_dict['sg_sd'] = p[2][1]
        p_dict['sg_sg'] = p[2][2]
        p_dict['sg_bg'] = p[2][3]
        p_dict['bg_bd'] = p[3][0]
        p_dict['bg_sd'] = p[3][1]
        p_dict['bg_sg'] = p[3][2]
        p_dict['bg_bg'] = p[3][3]

    elif (trend_length == 3):
        p_dict['bd_bd_bd'] = p[0][0]
        p_dict['bd_bd_sd'] = p[0][1]
        p_dict['bd_bd_sg'] = p[0][2]
        p_dict['bd_bd_bg'] = p[0][3]
        p_dict['bd_sd_bd'] = p[1][0]
        p_dict['bd_sd_sd'] = p[1][1]
        p_dict['bd_sd_sg'] = p[1][2]
        p_dict['bd_sd_bg'] = p[1][3]
        p_dict['bd_sg_bd'] = p[2][0]
        p_dict['bd_sg_sd'] = p[2][1]
        p_dict['bd_sg_sg'] = p[2][2]
        p_dict['bd_sg_bg'] = p[2][3]
        p_dict['bd_bg_bd'] = p[3][0]
        p_dict['bd_bg_sd'] = p[3][1]
        p_dict['bd_bg_sg'] = p[3][2]
        p_dict['bd_bg_bg'] = p[3][3]
        
        p_dict['sd_bd_bd'] = p[4][0]
        p_dict['sd_bd_sd'] = p[4][1]
        p_dict['sd_bd_sg'] = p[4][2]
        p_dict['sd_bd_bg'] = p[4][3]
        p_dict['sd_sd_bd'] = p[5][0]
        p_dict['sd_sd_sd'] = p[5][1]
        p_dict['sd_sd_sg'] = p[5][2]
        p_dict['sd_sd_bg'] = p[5][3]
        p_dict['sd_sg_bd'] = p[6][0]
        p_dict['sd_sg_sd'] = p[6][1]
        p_dict['sd_sg_sg'] = p[6][2]
        p_dict['sd_sg_bg'] = p[6][3]
        p_dict['sd_bg_bd'] = p[7][0]
        p_dict['sd_bg_sd'] = p[7][1]
        p_dict['sd_bg_sg'] = p[7][2]
        p_dict['sd_bg_bg'] = p[7][3]
        
        p_dict['sg_bd_bd'] = p[8][0]
        p_dict['sg_bd_sd'] = p[8][1]
        p_dict['sg_bd_sg'] = p[8][2]
        p_dict['sg_bd_bg'] = p[8][3]
        p_dict['sg_sd_bd'] = p[9][0]
        p_dict['sg_sd_sd'] = p[9][1]
        p_dict['sg_sd_sg'] = p[9][2]
        p_dict['sg_sd_bg'] = p[9][3]
        p_dict['sg_sg_bd'] = p[10][0]
        p_dict['sg_sg_sd'] = p[10][1]
        p_dict['sg_sg_sg'] = p[10][2]
        p_dict['sg_sg_bg'] = p[10][3]
        p_dict['sg_bg_bd'] = p[11][0]
        p_dict['sg_bg_sd'] = p[11][1]
        p_dict['sg_bg_sg'] = p[11][2]
        p_dict['sg_bg_bg'] = p[11][3]
        
        p_dict['bg_bd_bd'] = p[12][0]
        p_dict['bg_bd_sd'] = p[12][1]
        p_dict['bg_bd_sg'] = p[12][2]
        p_dict['bg_bd_bg'] = p[12][3]
        p_dict['bg_sd_bd'] = p[13][0]
        p_dict['bg_sd_sd'] = p[13][1]
        p_dict['bg_sd_sg'] = p[13][2]
        p_dict['bg_sd_bg'] = p[13][3]
        p_dict['bg_sg_bd'] = p[14][0]
        p_dict['bg_sg_sd'] = p[14][1]
        p_dict['bg_sg_sg'] = p[14][2]
        p_dict['bg_sg_bg'] = p[14][3]
        p_dict['bg_bg_bd'] = p[15][0]
        p_dict['bg_bg_sd'] = p[15][1]
        p_dict['bg_bg_sg'] = p[15][2]
        p_dict['bg_bg_bg'] = p[15][3]

    else:
        raise ValueError('Only trend lengths of 1-3 are supported right now')

    return p_dict

class OneDayModel:
    trained=False
    verbose=False
    probabilities = {}
    
    def __init__(self, data=None):
        if self.verbose: print('Initializing Model...')
        if data is not None:
            self.train(data)
        else:
            print('Inintializing without training data')
        
    def __str__(self):
        return 'One Day Model'

    def train(self, movement_categories):
        if self.verbose: print('Training...')
        p = get_single_day_probabilities(movement_categories)
        self.probabilities = p_array_to_dict(p, 1)
        self.trained=True
        
    def predict(self, input_sequence):
        predictions = []
        p_labels = []
        p_vals = []
        
        for p_label, p_val in self.probabilities.items():
            p_labels.append(p_label)
            p_vals.append(p_val)
        for i in range(len(input_sequence)):
            predictions.append(choose_category(p_labels,p_vals))
        return predictions

class TwoDayModel:
    trained=False
    verbose=False
    probabilities = {}
    input_to_possibile_outs = {'bd': ['bd_bd', 'bd_sd', 'bd_sg', 'bd_bg'],
                               'sd': ['sd_bd', 'sd_sd', 'sd_sg', 'sd_bg'],
                               'sg': ['sg_bd', 'sg_sd', 'sg_sg', 'sg_bg'],
                               'bg': ['bg_bd', 'bg_sd', 'bg_sg', 'bg_bg']}

    def __init__(self, data=None):
        if self.verbose: print('Initializing Model...')
        if data is not None:
            self.train(data)
        else:
            print('Inintializing without training data')
            
    def __str__(self):
        return 'Local Two Day Model'

    def train(self, movement_categories):
        if self.verbose: print('Training...')
        two_day_trends = get_trends(movement_categories, 2)
        p = build_model_probabilities(movement_categories, two_day_trends, 2)
        self.probabilities = p_array_to_dict(p, 2)
        self.trained=True
        
    def predict(self, input_sequence, raw=False):
        predictions = []
        p_labels = []
        p_vals = []
            
        for i in range(len(input_sequence)):
            #print(input_sequence[i])
            p_labels = self.input_to_possibile_outs[input_sequence[i]]
            #print(p_labels)
            for i in range(len(p_labels)):
                p_vals.append(self.probabilities[p_labels[i]])
            raw_choice = choose_category(p_labels, p_vals) ## input_output
            choice = raw_choice.rsplit('_', 1)[1] ## output
            if raw:
                predictions.append(raw_choice)
            else:
                predictions.append(choice)
        return predictions

class TwoDayCompositeModel:
    trained=False
    verbose=False
    probabilities = {}
    input_to_possibile_outs = {'bd': ['bd_bd', 'bd_sd', 'bd_sg', 'bd_bg'],
                               'sd': ['sd_bd', 'sd_sd', 'sd_sg', 'sd_bg'],
                               'sg': ['sg_bd', 'sg_sd', 'sg_sg', 'sg_bg'],
                               'bg': ['bg_bd', 'bg_sd', 'bg_sg', 'bg_bg']}

    def __init__(self):
        self.train()

    def __str__(self):
        return 'Composite Two Day Model'
            
    def train(self):
        (train_all_two_day_trends, _, _, train_all_movement_categories, _, _, _, _) = \
                get_train_valid_trends_all_stocks(1, 2, movement_category_types, n_cats=4)
        
        train_composite_two_day_probs = build_model_probabilities(train_all_movement_categories, 
                                                                  train_all_two_day_trends, 2)
        p = train_composite_two_day_probs
        self.probabilities = p_array_to_dict(p, 2)
        self.trained=True

    def predict(self, input_sequence, raw=False):
        predictions = []
        p_labels = []
        p_vals = []
            
        for i in range(len(input_sequence)):
            #print(input_sequence[i])
            p_labels = self.input_to_possibile_outs[input_sequence[i]]
            #print(p_labels)
            for j in range(len(p_labels)):
                p_vals.append(self.probabilities[p_labels[j]])
            raw_choice = choose_category(p_labels, p_vals) ## input_output
            choice = raw_choice.rsplit('_', 1)[1] ## output
            if raw:
                predictions.append(raw_choice)
            else:
                predictions.append(choice)
        return predictions

class TwoDayVolumeModel:
    trained=False
    verbose=False
    probabilities = {}
    input_to_possibile_outs = {'vl': ['vl_bd', 'vl_sd', 'vl_sg', 'vl_bg'],
                               'l':  ['l_bd', 'l_sd', 'l_sg', 'l_bg'],
                               'h':  ['h_bd', 'h_sd', 'h_sg', 'h_bg'],
                               'vh': ['vh_bd', 'vh_sd', 'vh_sg', 'vh_bg']}

    def __init__(self, movement_categories=None, volume_categories=None):
        if self.verbose: print('Initializing Model...')
        if movement_categories is not None and volume_categories is not None:
            self.train(movement_categories, volume_categories)
        else:
            print('Inintializing without training data')
       
    def __str__(self):
        return 'Two Day Volume Model'

    def train(self, movement_categories, volume_categories):
        if self.verbose: print('Training...')
        two_day_volume_trends = get_two_day_volume_trends(volume_categories, movement_categories)
        #print(two_day_volume_trends[0:30])
        p = build_model_probabilities(movement_categories, two_day_volume_trends, 
                                      2, previous_category_types=['vl', 'l', 'h', 'vh'])
        self.probabilities['vl_bd'] = p[0][0]
        self.probabilities['vl_sd'] = p[0][1]
        self.probabilities['vl_sg'] = p[0][2]
        self.probabilities['vl_bg'] = p[0][3]
        self.probabilities['l_bd'] = p[1][0]
        self.probabilities['l_sd'] = p[1][1]
        self.probabilities['l_sg'] = p[1][2]
        self.probabilities['l_bg'] = p[1][3]
        self.probabilities['h_bd'] = p[2][0]
        self.probabilities['h_sd'] = p[2][1]
        self.probabilities['h_sg'] = p[2][2]
        self.probabilities['h_bg'] = p[2][3]
        self.probabilities['vh_bd'] = p[3][0]
        self.probabilities['vh_sd'] = p[3][1]
        self.probabilities['vh_sg'] = p[3][2]
        self.probabilities['vh_bg'] = p[3][3]
        self.trained=True
        
    def predict(self, input_sequence, raw=False):
        predictions = []
        p_labels = []
        p_vals = []
            
        for i in range(len(input_sequence)):
            p_labels = self.input_to_possibile_outs[input_sequence[i]]
            for j in range(len(p_labels)):
                p_vals.append(self.probabilities[p_labels[j]])
            raw_choice = choose_category(p_labels, p_vals) ## input_output
            choice = raw_choice.rsplit('_', 1)[1] ## output
            if raw:
                predictions.append(raw_choice)
            else:
                predictions.append(choice)
        return predictions

class ThreeDayModel:
    trained=False
    verbose=False
    probabilities = {}
    input_to_possibile_outs = {'bd_bd': ['bd_bd_bd', 'bd_bd_sd', 'bd_bd_sg', 'bd_bd_bg'],
                               'bd_sd': ['bd_sd_bd', 'bd_sd_sd', 'bd_sd_sg', 'bd_sd_bg'],
                               'bd_sg': ['bd_sg_bd', 'bd_sg_sd', 'bd_sg_sg', 'bd_sg_bg'],
                               'bd_bg': ['bd_bg_bd', 'bd_bg_sd', 'bd_bg_sg', 'bd_bg_bg'],
                               
                               'sd_bd': ['sd_bd_bd', 'sd_bd_sd', 'sd_bd_sg', 'sd_bd_bg'],
                               'sd_sd': ['sd_sd_bd', 'sd_sd_sd', 'sd_sd_sg', 'sd_sd_bg'],
                               'sd_sg': ['sd_sg_bd', 'sd_sg_sd', 'sd_sg_sg', 'sd_sg_bg'],
                               'sd_bg': ['sd_bg_bd', 'sd_bg_sd', 'sd_bg_sg', 'sd_bg_bg'],
                               
                               'sg_bd': ['sg_bd_bd', 'sg_bd_sd', 'sg_bd_sg', 'sg_bd_bg'],
                               'sg_sd': ['sg_sd_bd', 'sg_sd_sd', 'sg_sd_sg', 'sg_sd_bg'],
                               'sg_sg': ['sg_sg_bd', 'sg_sg_sd', 'sg_sg_sg', 'sg_sg_bg'],
                               'sg_bg': ['sg_bg_bd', 'sg_bg_sd', 'sg_bg_sg', 'sg_bg_bg'],
                               
                               'bg_bd': ['bg_bd_bd', 'bg_bd_sd', 'bg_bd_sg', 'bg_bd_bg'],
                               'bg_sd': ['bg_sd_bd', 'bg_sd_sd', 'bg_sd_sg', 'bg_sd_bg'],
                               'bg_sg': ['bg_sg_bd', 'bg_sg_sd', 'bg_sg_sg', 'bg_sg_bg'],
                               'bg_bg': ['bg_bg_bd', 'bg_bg_sd', 'bg_bg_sg', 'bg_bg_bg']}
    
    def __init__(self, data=None):
        if self.verbose: print('Initializing Model...')
        if data is not None:
            self.train(data)
        else:
            print('Inintializing without training data')
            
    def __str__(self):
        return 'Three Day Model'

    def train(self, movement_categories):
        if self.verbose: print('Training...')
        three_day_trends = get_trends(movement_categories, 3)
        p = build_model_probabilities(movement_categories, three_day_trends, 3)
        self.probabilities = p_array_to_dict(p, 3)
        self.trained=True
        
    def predict(self, input_sequence, raw=False):
        predictions = []
        p_labels = []
        p_vals = []
            
        for i in range(len(input_sequence) - 1):
            p_labels = self.input_to_possibile_outs[input_sequence[i] + '_' + input_sequence[i+1]]
            for j in range(len(p_labels)):
                p_val = self.probabilities[p_labels[j]]
                p_vals.append(p_val)
            raw_choice = choose_category(p_labels, p_vals) ## input_output
            choice = raw_choice.rsplit('_')[2] ## output
            if raw:
                predictions.append(raw_choice)
            else:
                predictions.append(choice)
        return predictions


##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
## Nb 18: MACD and Moving average stuff
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
def get_close_price(df):
    """ Get the simple moving average of a stock that's in a data frame. """
    df = df.sort_index(axis=0) ## We want the dates in ascending order
    close = np.zeros(len(df))

    for i in range(len(df)):
        close[i] = df['close'][i]

    return close

def get_sma(df, period_length):
    """ Get the simple moving average of a stock that's in a data frame. """
    #df = df.sort_index(axis=0) ## We want the dates in ascending order
    sma = np.zeros(len(df))
    sum_period = np.zeros(len(df))
    count = 0

    for i in range(len(df)):
        count = 0
        for j in range(period_length):
            if (i - j) >= 0:
                sum_period[i] += df[i - j]
                count += 1
        sma[i] = sum_period[i] / count
    return sma

def get_ema(df, period_length, alpha):
    """ Get the explonential moving average of a stock that's in a data frame. """
    #df = df.sort_index(axis=0) ## We want the dates in ascending order
    ema = np.zeros(len(df))
    sum_period = np.zeros(len(df))
    count = 0
    if not (0 <= alpha <= 1):
        raise ValueError('Alpha should be between 0 and 1')

    for i in range(len(df)):
        count = 0
        this_alpha = alpha
        for j in range(period_length):
            if (i - j) >= 0:
                sum_period[i] += df[i - j] * this_alpha
                count += this_alpha
                #print(this_alpha / alpha)
                this_alpha *= alpha
        ema[i] = sum_period[i] / count
    return ema

def plot_macd(fast_period, slow_period, macd_ema_period, viewing_window, 
              alpha=0.95, show_ema=False, show_early_signals=False, show_macd_chart=True, 
              figsize=(16,6), ticker=None, threshold=0):    
    if ticker is None:
        close = get_close_price(df)
    else:
        df = pd.DataFrame()
        df = df.from_csv('stock_data/' + ticker.lower() +'.csv')
        df = df.sort_index(axis=0)
        close = get_close_price(df)
    fast_leg = get_ema(close, fast_period, alpha)
    slow_leg = get_ema(close, slow_period, alpha)
    macd = fast_leg - slow_leg
    macd_signal = get_ema(macd, macd_ema_period, alpha)
    macd_diff = macd - macd_signal
    
    zero_line = np.zeros(len(macd))
    
    buy_signals = []
    sell_signals = []
    early_buy_signals = []
    early_sell_signals = []

    for i in range(len(macd_diff) - 1):
        if macd_diff[i] < threshold and macd_diff[i+1] > threshold:
            buy_signals.append((i, close[i]))
        elif macd_diff[i] > -threshold and macd_diff[i+1] < -threshold:
            sell_signals.append((i, close[i]))
            
    for i in range(len(macd_diff) - 1):
        if macd_diff[i] < 0 and  macd_diff[i] < macd_diff[i+1]:
            early_buy_signals.append((i, close[i]))
        elif macd_diff[i] > 0 and macd_diff[i] > macd_diff[i+1]:
            early_sell_signals.append((i, close[i]))
    
    plt.figure(figsize=figsize)
    plt.plot(close[-viewing_window:], label='Stock Price')
    if show_ema:
        plt.plot(fast_leg[-viewing_window:], label='Fast EMA')
        plt.plot(slow_leg[-viewing_window:], label='Slow EMA')
        
    plt.title('Stock Price and Signals')
    
    if show_early_signals:
        for i in range(len(early_buy_signals)):
            if early_buy_signals[i][0] > len(macd) - viewing_window:
                plt.scatter(early_buy_signals[i][0] - (len(macd) - viewing_window), 
                            early_buy_signals[i][1], c='#8fba80')
        for i in range(len(early_sell_signals)):
            if early_sell_signals[i][0] > len(macd) - viewing_window:
                plt.scatter(early_sell_signals[i][0] - (len(macd) - viewing_window), 
                early_sell_signals[i][1], c='#c19b95')##c1897f
    
    for i in range(len(buy_signals)):
        if buy_signals[i][0] > len(macd) - viewing_window:
            plt.scatter(buy_signals[i][0] - (len(macd) - viewing_window), buy_signals[i][1], c='green')
    for i in range(len(sell_signals)):
        if sell_signals[i][0] > len(macd) - viewing_window:
            plt.scatter(sell_signals[i][0] - (len(macd) - viewing_window), sell_signals[i][1], c='red')
        
    plt.legend()
    
    ## MACD
    if show_macd_chart:
        plt.figure(figsize=figsize)
        plt.plot(zero_line[-viewing_window:], c='black')
        plt.plot(macd[-viewing_window:], label='MACD', c='blue')
        plt.plot(macd_signal[-viewing_window:], label='MACD Signal Line', c='green')
        plt.plot(macd_diff[-viewing_window:], label='MACD Difference', c='red', linestyle='dashed')
        plt.title('MACD')
        plt.legend()
    plt.show()




