import matplotlib.patches as mpatches

def plot_three_day_probability_bar_graph(previous_day, two_day_trends, three_day_trends):
	two_day_probs = []
	three_day_probs = []
	all_categories = ['bd', 'sd', 'sg', 'bg']
	count = count_movement_category(movement_categories, previous_day)

    ## Get probabilities after 'previous_day'
	for next_day in all_categories:
		two_day_name = previous_day +'_' + next_day
		two_day_count = count_two_day_trends(two_day_trends, two_day_name)
		two_day_prob = two_day_count / count
		two_day_probs.append(two_day_prob)
    
    ## Get probabilities after 'previous_day' and the day before
	for next_day in all_categories:
		for day_before_last in all_categories:  
			three_day_name = day_before_last +'_' + previous_day +'_' + next_day
			three_day_count = count_trends(three_day_trends, three_day_name)
			three_day_prob = three_day_count / count
			three_day_probs.append(three_day_prob)

	fig = plt.figure(figsize=(11,4))
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.9])
	categories = ('Big Drop', 'Small Drop', 'Small Gain', 'Big Gain')
	ind = np.arange(4)
	width = 0.2

    ## Plot three day probabilities
	for i in range(int(len(three_day_probs) / 4)):
		pl = ax.bar(ind[i] + width, three_day_probs[i * 4], width, color='red')
		height = three_day_probs[i * 4]
		pl = ax.bar(ind[i] + width, three_day_probs[i * 4 + 1], width, color='orange', bottom=height)
		height += three_day_probs[i * 4 + 1]
		pl = ax.bar(ind[i] + width, three_day_probs[i * 4 + 2], width, color='#ebfaeb', bottom=height)
		height += three_day_probs[i * 4 + 2]
		pl = ax.bar(ind[i] + width, three_day_probs[i * 4 + 3], width, color='#6efa70', bottom=height)

    ## Plot two day probability 
	conditioned_pl = ax.bar(ind + (2 * width), two_day_probs, width, color='b')
        
    
    ## TODO legend
	labels = ['After a ' + category_full_names[previous_day], 
              'After a big drop, then a ' + category_full_names[previous_day],
              'After a small drop, then a ' + category_full_names[previous_day], 
              'After a small gain, then a ' + category_full_names[previous_day], 
              'After a big gain, then a ' + category_full_names[previous_day]]
	composite_patch = mpatches.Patch(color='blue')
	bd_x_patch = mpatches.Patch(color='red')
	sd_x_patch = mpatches.Patch(color='orange')
	sg_x_patch = mpatches.Patch(color='#ebfaeb')
	bg_x_patch = mpatches.Patch(color='#6efa70')
	fig.legend([composite_patch, bd_x_patch, sd_x_patch, sg_x_patch, bg_x_patch], labels, 'upper right')
	ax.text(0.5, max(two_day_probs) * .95, 'n = ' + '{0:d}'.format(count), ha='center', va='center', weight='medium')

	plt.ylabel('Probabilities')
	plt.title('Probabilities of each Category')
	plt.xticks(ind + 2 * width, categories)
	plt.legend()