from django.http import HttpResponse
from django.shortcuts import render_to_response
from .helper import *
from keras.models import load_model
from .settings import BASE_DIR
import numpy as np
import tensorflow as tf
from django.shortcuts import HttpResponseRedirect


# Main Home screen
def home(request):
	global teams
	teams = list(get_two_random_teams().keys())
	return render_to_response("index.html",{'team1':teams[0], 'team2':teams[1]}) # We will pass random values of the teams to show on every reload of the page.


# Returns the about.html page
def about(request):
	return render_to_response("about.html") 

# Returns the list of leaderboard if any	
def leaderboards(request):
	return render_to_response("leaderboard.html") 

def predict(request):
	global teams
	win_team_acc_to_human = request.GET.get("winning_team","") # Will have to perform mapping to get the value
	toss_decision = request.GET.get("toss_decision") # 1 stands for toss_decision True
	team1 = request.GET.get("team1","")
	team2 = request.GET.get("team2","")

	index = mapping.get(team1)
	index2 = mapping.get(team2)

	model = return_model()
	t_decision = None
	if toss_decision == team1 or toss_decision == team2:
		t_decision = 1
	else:
		t_decision = 0


	x = np.array([ [index, index2, t_decision] ])

	pred = model.predict(x)
	pred_index = np.argmax(pred, axis=1)[0]

	print (win_team_acc_to_human, mapping.get(pred_index))
	if win_team_acc_to_human == mapping.get(pred_index):
		return HttpResponse("<script> alert('You WIN!'); </script>")
		return HttpResponseRedirect("/home/")
	else:
		return HttpResponse("<script> alert('You LOST!'); </script>")


	#print (index, index2)

