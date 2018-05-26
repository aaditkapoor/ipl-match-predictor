# Constants in the program
import random
from keras.models import load_model
import numpy as np
from .settings import BASE_DIR


mapping = {'Chennai Super Kings': 8,
 'Deccan Chargers': 10,
 'Delhi Daredevils': 6,
 'Gujarat Lions': 2,
 'Kings XI Punjab': 7,
 'Kochi Tuskers Kerala': 11,
 'Kolkata Knight Riders': 5,
 'Mumbai Indians': 1,
 'Pune Warriors': 12,
 'Rajasthan Royals': 9,
 'Rising Pune Supergiant': 3,
 'Rising Pune Supergiants': 13,
 'Royal Challengers Bangalore': 4,
 'Sunrisers Hyderabad': 0}

# Loading the model
def return_model():
	model = load_model(BASE_DIR+"/predictor/model/match-predictor.h5")
	return model

# Returns two random teams with their respective labels
def get_two_random_teams():
	global mapping

	option1 = random.randint(0,13)
	option2 = random.randint(0,13)

	team1 = ""
	team2 = ""


	for i in mapping.items():
		if i[1] == option1:
			team1 = i[0]
		if i[1] == option2:
			team2 = i[0]

	random_data = { team1 : mapping.get(team1), team2:mapping.get(team2) }
	return random_data


# Returns the predicted winner according to the machine learning model.
def get_prediction(team1, team2, toss_decision):
	global model
	global mapping

	team1 = mapping.get(team1)
	team2 = mapping.get(team2)
	if toss_decision:
		toss_decision = 1
	else:
		toss_decision = 0

	data = np.array([[ team1, team2, toss_decision] ]) # Changing into 2 dimensional for prediciton

	# Prediciton begins
	pred = np.argmax(model.predict(data), axis=1)

	# Getting the team from the number
	return mapping.get(pred)


# Checks if human decision is same as computer's.
def human_did_win(team1, team2, toss_decision, human_decision):
	computer_decision = get_prediction(team1, team2, toss_decision)
	if human_decision == computer_decision:
		return True # Human wins
		# Maybe save the result
	else:
		return False # Computer wins


