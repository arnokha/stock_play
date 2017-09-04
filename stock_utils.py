## Commonly used functions in my exercises

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats


category_full_names = {'vbd':'very big drop', 'bd':'big drop', 'md':'medium drop', 'sd':'small drop',
                       'vbg':'very big gain', 'bg':'big gain', 'mg':'medium gain', 'sg':'small gain'}

def ticker_from_csv(csv_string):
	""" 
	The downloaded files come in the form [ticker].csv. 
	We are just stripping off the csv extension and making the ticker uppercase.
	"""
	stock_name = csv_string.rsplit('.', 1)[0] ## Peel off the ".csv" from the given string
	return stock_name.upper()

def get_price_movements(df, period=1):
	""" Get the movement of a stock that's in a data frame. """
	df = df.sort_index(axis=0) ## We want the dates in ascending order
	movement = np.zeros(int(len(df) / period)) ## Python's int function rounds down
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

def plot_gaussian(x, x_min=-10, x_max=10, n=10000, fill=False):
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
	plt.xlabel('Daily Movement Percentage')
	plt.ylabel('Density')

	if (fill):
		plt.fill_between(xs, 0, y)

def plot_gaussian_categorical(x, x_min=-10, x_max=10, n=10000, title=''):
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
	plt.fill_between(xs, 0, y, where= xs < mu, facecolor='#eeeedd', interpolate=True) ## Small Drop
	plt.fill_between(xs, 0, y, where= xs < (mu - sigma / 2), facecolor='yellow', interpolate=True) ## Medium Drop
	plt.fill_between(xs, 0, y, where= xs < (mu - sigma), facecolor='orange', interpolate=True) ## Big Drop
	plt.fill_between(xs, 0, y, where= xs < (mu - 2*sigma), facecolor='red', interpolate=True) ## Very big drop

	plt.fill_between(xs, 0, y, where= xs > mu, facecolor='#ebfaeb', interpolate=True) ## Small Gain
	plt.fill_between(xs, 0, y, where= xs > (mu + sigma/2), facecolor='#b5fbb6', interpolate=True) ## Gain
	plt.fill_between(xs, 0, y, where= xs > (mu + sigma), facecolor='#6efa70', interpolate=True) ## Big Gain
	plt.fill_between(xs, 0, y, where= xs > (mu + 2*sigma), facecolor='green', interpolate=True) ## Very Big Gain

	## Label mu and sigma
	plt.text(x_min + 1, max(y) * 0.8, r'$\mu$ = ' + '{0:.2f}'.format(mu))
	plt.text(x_min + 1, max(y) * 0.9, r'$\sigma$ = ' + '{0:.2f}'.format(sigma))
	## Set title if given
	if (len(title) != 0):
		plt.title(title)

def categorize_movements(movements, n_cats=8):
	"""
	Given an array of movements, return an array of categories based on how relatively large the movements are.
	The default number of categories is 8.
	"""
	mu, sigma = np.mean(movements), np.std(movements)
	categories = []

	if (n_cats == 8):
		for i in range(len(movements)):
			if (movements[i] <= (mu - 2*sigma)):
				categories.append('vbd') ## very big drop
			elif (movements[i] <= (mu - sigma)):
				categories.append('bd')  ## big drop
			elif (movements[i] <= (mu - sigma/2)):
				categories.append('md')  ## medium drop
			elif (movements[i] < mu):
				categories.append('sd')  ## small drop
			elif (movements[i] >= (mu + 2*sigma)):
				categories.append('vbg') ## very big gain
			elif (movements[i] >= (mu + sigma)):
				categories.append('bg')  ## big gain
			elif (movements[i] >= (mu + sigma/2)):
				categories.append('mg')  ## medium gain
			elif (movements[i] >= mu):
				categories.append('sg')  ## small gain

	elif (n_cats == 4):
		for i in range(len(movements)):
			if (movements[i] <= (mu - sigma)):
				categories.append('bd')  ## big drop
			elif (movements[i] < mu):
				categories.append('sd')  ## small drop
			elif (movements[i] >= (mu + sigma)):
				categories.append('bg')  ## big gain
			elif (movements[i] >= mu):
				categories.append('sg')  ## small gain

	else:
		raise ValueError('Currently only 4 and 8 categories are supported')

	return categories

def count_movement_category(categories, cat_to_count):
	count = 0
	for i in range(len(categories)):
		if categories[i] == cat_to_count:
			count = count + 1
	return count

def count_two_day_trends(trends, trend_to_count):
	raise NameError('Renamed to count_trends')

def count_trends(trends, trend_to_count):
	count = 0
	for i in range(len(trends)):
		if trends[i] == trend_to_count:
			count = count + 1
	return count

def get_two_day_trends(categories):
	two_day_trends = []
	for i in range(len(categories) - 1):
		two_day_trends.append(categories[i] + '_' + categories[i+1])
	return two_day_trends

def get_three_day_trends(categories):
	three_day_trends = []
	for i in range(len(categories) - 2):
		three_day_trends.append(categories[i] + '_' + categories[i+1] + '_' + categories[i+2])
	return three_day_trends

def plot_two_day_probability_bar_graph(previous_day, count, two_day_trends, cat_probs, n_cats=8, show_baseline=True):
	two_day_probs = []
	if (n_cats == 8):
		all_categories = ['vbd', 'bd', 'md', 'sd', 'sg', 'mg', 'bg', 'vbg']
	elif(n_cats == 4):
		all_categories = ['bd', 'sd', 'sg', 'bg']
	for next_day in all_categories:
		two_day_name = previous_day +'_' + next_day
		two_day_count = count_two_day_trends(two_day_trends, two_day_name)
		two_day_prob = two_day_count / count
		two_day_probs.append(two_day_prob)

	plt.figure(figsize=(11,4))
	if (n_cats == 8):
		categories = ('Very Big Drop', 'Big Drop', 'Medium Drop', 'Small Drop', 'Small Gain', 'Medium Gain', 'Big Gain', 'Very Big Gain')
		ind = np.arange(8)
	elif (n_cats == 4):
		categories = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
		ind = np.arange(4)
	width = 0.25

	if (show_baseline):
		orig_pl = plt.bar(ind+width, cat_probs, width, color='b', label='Original')
	conditioned_pl = plt.bar(ind, two_day_probs, width, color='r', label='After a ' + category_full_names[previous_day])

	plt.text(0.5, max(two_day_probs) * .95, 'n = ' + '{0:d}'.format(count), ha='center', va='center', weight='medium')

	plt.ylabel('Probabilities')
	plt.title('Probabilities of each Category')
	plt.xticks(ind+width, categories)
	plt.legend()
	#plt.show()
