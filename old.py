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
		
	## Very unlikely to happen, but will catch in the case none of the samples have the category we're looking for
	return 0

def get_probabilities(two_day_trends, categories, n_categories=4):
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

def get_three_day_probabilities(three_day_trends, two_day_name, categories, n_categories=4):
	"""Returns the probability distribution for the third day given the previous two"""
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
	two_day_name = categorize_movement(step_before_last, mu, sigma) + '_' + categorize_movement(previous_step, mu, sigma)
	choice = choose_category(['bd', 'sd', 'sg', 'bg'], three_day_probability)
	random_samples = np.random.normal(mu, sigma, 1000)
	
	## Draw on random samples until we get a result of the correct category
	for i in range(len(random_samples)):
		if (categorize_movement(random_samples[i], mu, sigma) == choice):
			return random_samples[i]
		
	## Very unlikely to happen, but will catch in the case none of the samples have the category we're looking for
	return 0

def run_three_day_momentum_simulation(prior_daily_movements, starting_value, mu, sigma, n_steps, n_trials):
	## Get categories and trends
	prior_movement_categories = categorize_movements(prior_daily_movements, n_cats=4)
	prior_three_day_trends = get_three_day_trends(prior_movement_categories)
	
	trials = []
	
	## Collect a dictionay of three day probabilities
	all_categories = ['bd', 'sd', 'sg', 'bg']
	three_day_probs = {}
	for first_day in all_categories:
		for next_day in all_categories:
			two_day_name = first_day + '_' + next_day
			three_day_probs[two_day_name] = get_three_day_probabilities(prior_three_day_trends, two_day_name, prior_movement_categories)
	
	## Generate steps based on the movements of the prior two days
	for i in range(n_trials):
		two_day_name = categorize_movement(prior_daily_movements[-2], mu, sigma) + '_' + categorize_movement(prior_daily_movements[-1], mu, sigma)
		three_day_prob = three_day_probs[two_day_name]
		first_step = generate_next_three_day_step(prior_daily_movements[-2], prior_daily_movements[-1], three_day_prob, mu, sigma)
		
		two_day_name = categorize_movement(prior_daily_movements[-2], mu, sigma) + '_' + categorize_movement(prior_daily_movements[-1], mu, sigma)
		three_day_prob = three_day_probs[two_day_name]
		second_step = generate_next_three_day_step(prior_daily_movements[-1], first_step, three_day_prob, mu, sigma)
		
		steps = [first_step, second_step]

		for i in range(n_steps - 2):
			two_day_name = categorize_movement(steps[i], mu, sigma) + '_' + categorize_movement(steps[i+1], mu, sigma)
			three_day_prob = three_day_probs[two_day_name]
			steps.append(generate_next_three_day_step(steps[i], steps[i+1], three_day_prob, mu, sigma))
		
		trials.append(simulate_movements(steps, starting_value))

    g = glob.glob('stock_data/*.csv')

all_weekly_movements = []
all_weekly_categories = []
all_two_week_trends = []

w_vbd_count = 0
w_bd_count = 0
w_md_count = 0
w_sd_count = 0
w_sg_count = 0
w_mg_count = 0
w_bg_count = 0
w_vbg_count = 0
w_total_cat_count = 0

for i in range(len(g)):
    df = pd.DataFrame()
    df = df.from_csv(g[i])
    weekly_movements = get_price_movements(df, period=7)
    weekly_categories = categorize_movements(weekly_movements)
    
    all_weekly_movements.extend(weekly_movements)
    all_weekly_categories.extend(weekly_categories)
    
    w_vbd_count += count_movement_category(weekly_categories, 'vbd')
    w_bd_count += count_movement_category(weekly_categories, 'bd')
    w_md_count += count_movement_category(weekly_categories, 'md')
    w_sd_count += count_movement_category(weekly_categories, 'sd')
    w_sg_count += count_movement_category(weekly_categories, 'sg')
    w_mg_count += count_movement_category(weekly_categories, 'mg')
    w_bg_count += count_movement_category(weekly_categories, 'bg')
    w_vbg_count += count_movement_category(weekly_categories, 'vbg')
    w_total_cat_count += len(weekly_categories)
    
    two_week_trends = get_trends(weekly_categories, 2)
    all_two_week_trends.extend(two_week_trends)

w_p_vbd = w_vbd_count / w_total_cat_count
w_p_bd = w_bd_count / w_total_cat_count
w_p_md = w_md_count / w_total_cat_count
w_p_sd = w_sd_count / w_total_cat_count
w_p_sg = w_sg_count / w_total_cat_count
w_p_mg = w_mg_count / w_total_cat_count
w_p_bg = w_bg_count / w_total_cat_count
w_p_vbg = w_vbd_count / w_total_cat_count

w_cat_counts = [w_vbd_count, w_bd_count, w_md_count, w_sd_count, w_sg_count, w_mg_count, w_bg_count, w_vbg_count]
w_cat_probs = [w_p_vbd, w_p_bd, w_p_md, w_p_sd, w_p_sg, w_p_mg, w_p_bg, w_p_vbg]

	return trials