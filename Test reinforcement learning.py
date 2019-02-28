#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import random
import itertools
from IPython.display import clear_output
from gym import utils
from gym.envs.toy_text import discrete
import pandas as pd
import pymongo
from pymongo import MongoClient
from collections import defaultdict
import matplotlib
import sys

#Variabili DB
connection = MongoClient('localhost', 27017)
db = connection.Sensors
collection = db.steps_bmi
training_collection = db.training_set
test_collection = db.test_set

db_env = connection.Environments
environment_collection = db_env.env03
#env01: 5-7gg, no max reward, incremento delta bmi superiore all'1%, taglio 5% code
#env02 7-10gg, no max reward, incremento delta bmi superiore all'1%, taglio 5% code
#env03 5-7gg, no max reward, incremento unitario per variazioni di bmi superiore all'1%, taglio 5% code
#env04 7-10gg, no max reward, incremento unitario per variazioni di bmi superiore all'1%, taglio 5% code


#Variabili per il windowing dei dati
minimum_bmi_interval = 5
maximum_bmi_interval = 7
days_interval = 6


#Variabili per il binning
num_bins_steps = 5
num_bins_bmi = 5

#Variabili per il cutting dei dati
lower_percentile = 5
higher_percentile = 95

#Variabili per il reinforcement Learning
max_reward = 1000
min_reward = -1000

#Variabili threshold
min_bmi_variation_percentage  = 1
weekly_actions_threshold = 0.5
min_significative_steps = 1000
#recommendation_threshold = 0.7

#max_users = (len(training_collection.find().distinct("User_id")) // 3)*2
#print(max_users)


# In[2]:


#Take a list of Object with "Date" attribute (d-m-y format) and return the list ordered by date
def mergeSortByDate(dated_list):
    #print("Splitting ",dateList)
    if len(dated_list)>1:
        mid = len(dated_list)//2
        lefthalf = dated_list[:mid]
        righthalf = dated_list[mid:]

        mergeSortByDate(lefthalf)
        mergeSortByDate(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            ymd_left_half = lefthalf[i]["Date"].split("-")
            ymd_right_half = righthalf[j]["Date"].split("-")
            if(ymd_left_half[0] == ymd_right_half[0]):
                if(ymd_left_half[1] == ymd_right_half[1]):
                    if(int(ymd_left_half[2]) < int(ymd_right_half[2])):
                        dated_list[k]=lefthalf[i]
                        i=i+1
                    else:
                        dated_list[k]=righthalf[j]
                        j=j+1
                else:
                    if(int(ymd_left_half[1]) < int(ymd_right_half[1])):
                        dated_list[k]=lefthalf[i]
                        i=i+1
                    else:
                        dated_list[k]=righthalf[j]
                        j=j+1
            else:
                if(int(ymd_left_half[0]) < int(ymd_right_half[0])):
                    dated_list[k]=lefthalf[i]
                    i=i+1
                else:
                    dated_list[k]=righthalf[j]
                    j=j+1
            k=k+1

        while i < len(lefthalf):
            dated_list[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            dated_list[k]=righthalf[j]
            j=j+1
            k=k+1
    #print("Merging ",dateList)
    
    return dated_list

def mergeSortByValue(dated_list):
    if len(dated_list)>1:
        mid = len(dated_list)//2
        lefthalf = dated_list[:mid]
        righthalf = dated_list[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i]["Value"] < righthalf[j]["Value"]:
                dated_list[k]=lefthalf[i]
                i=i+1
            else:
                dated_list[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            dated_list[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            dated_list[k]=righthalf[j]
            j=j+1
            k=k+1
            
    return dated_list
    

#Take a list of Dates and return the list sorted
def mergeSortDates(dateList):
    #print("Splitting ",dateList)
    if len(dateList)>1:
        mid = len(dateList)//2
        lefthalf = dateList[:mid]
        righthalf = dateList[mid:]

        mergeSortDates(lefthalf)
        mergeSortDates(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            ymd_left_half = lefthalf[i].split("-")
            ymd_right_half = righthalf[j].split("-")
            if(ymd_left_half[0] == ymd_right_half[0]):
                if(ymd_left_half[1] == ymd_right_half[1]):
                    if(int(ymd_left_half[2]) < int(ymd_right_half[2])):
                        dateList[k]=lefthalf[i]
                        i=i+1
                    else:
                        dateList[k]=righthalf[j]
                        j=j+1
                else:
                    if(int(ymd_left_half[1]) < int(ymd_right_half[1])):
                        dateList[k]=lefthalf[i]
                        i=i+1
                    else:
                        dateList[k]=righthalf[j]
                        j=j+1
            else:
                if(int(ymd_left_half[0]) < int(ymd_right_half[0])):
                    dateList[k]=lefthalf[i]
                    i=i+1
                else:
                    dateList[k]=righthalf[j]
                    j=j+1
            k=k+1

        while i < len(lefthalf):
            dateList[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            dateList[k]=righthalf[j]
            j=j+1
            k=k+1
    #print("Merging ",dateList)
    return dateList


def extractValues(object_list):
    values_list = []
    for obj in object_list:
        value = obj['Value']
        values_list.append(value)
    return values_list


def getValueByDate(dated_values_list, date):
    for obj in dated_values_list:
        if(obj["Date"] == date):
            return obj["Value"]
    return -1


#Take a list of dated values objects and return the indices of valid values beetween a minimum and maximum date interval
def getValidIndices(dated_values_list, starting_window_index, dates):
    if((starting_window_index + minimum_bmi_interval) > len(dates)):
        return -1
    else:
        done = False
        while not done:
            while getValueByDate(dated_values_list, dates[starting_window_index]) == -1:
                starting_window_index += 1
                if((starting_window_index + minimum_bmi_interval) > len(dates)):
                    return -1

            ending_window_index = starting_window_index + (minimum_bmi_interval-1)

            while getValueByDate(dated_values_list, dates[ending_window_index]) == -1 and (ending_window_index - starting_window_index) < (maximum_bmi_interval-1):
                ending_window_index += 1
                if(ending_window_index >= len(dates)):
                    return -1

            if(getValueByDate(dated_values_list, dates[ending_window_index]) != -1):
                   done = True
            else:
                   starting_window_index += 1
    
    return (starting_window_index, ending_window_index)           
        
    
    
#Function that take a list of dated values and return the binned values    
def getBins(dated_values_list, num_bins, quartile = True):
    values = []
    date_bins_list = []
    for obj in dated_values_list:
        values.append(obj["Value"])
    if(quartile):
        labels = pd.qcut(values, num_bins, duplicates = "drop").codes
    else:
        labels = pd.cut(values, num_bins, duplicates = "drop").codes
    for i in range(len(dated_values_list)):
        date_bins_list.append({"Date": dated_values_list[i]["Date"], "Value": labels[i]})
    return date_bins_list
        

def getvaluesByDateInterval(dated_values_list, date_list):
    interval_values = []
    for date in date_list:
        value = getValueByDate(dated_values_list, date)
        if(value == -1):
            return -1
        else:
            interval_values.append(value)
    return interval_values


#Encode a stutus(a list of bins) in a number
def encode(state):
    code = 0
    for i in range(len(state)-1):
        code += state[i] 
        code *= num_bins_steps
    code += state[len(state)-1]
    return code


def decode(code):
    out = []
    for i in range(days_interval):
        out.append(code % num_bins_steps)
        code = code // num_bins_steps
    assert 0 <= code < num_bins_steps
    return out[::-1]



def nextState(encoded_state, action):
    next_state = []
    decoded_state = decode(encoded_state)
    for bin_ in decoded_state[1:]:
        next_state.append(bin_)
    next_state.append(action)
    return encode(next_state)



def getDates():
    dates = mergeSortDates(collection.find().distinct("Date"))
    return dates



def cutTails(dated_values_list):
    #print("Cutting")
    cutted_list = []
    values = np.array(extractValues(dated_values_list))
    lower = np.percentile(values, lower_percentile)
    higher = np.percentile(values, higher_percentile)
    for obj in dated_values_list:
        if(obj["Value"]> lower and obj["Value"]<higher):
            cutted_list.append(obj)
    return cutted_list



def retrieve(dataset, cleaning):
    print("Retrieving da ", dataset)
    users_steps = {}
    users_bmi = {}
    cursor = dataset.find()
    #cursor = collection.find({"User_id":  {"$in": [ "2bc16eda651db5936cd31e735c815296fc1579d9", "4b9150ccb6ca42e0aa43824c8ddcb28e326ab467", "ce96039ca66a9b98f97b202c177cf0ef7d4aa97d" ]}})
    
    for doc in cursor:
        #print(doc)
        user_id = doc["User_id"]
        value = doc["Value"]
        date = doc["Date"]
        data_type = doc["Data_type"]
        if(data_type == 1):
            if(cleaning == "minimum_steps"):
                
                if(value < min_significative_steps):
                    print("Valore sotto la soglia")
                    continue
        
            if(user_id in users_steps):
                users_steps[user_id].append({"Date": date, "Value": value})
            else: 
                users_steps[user_id] = [{"Date": date, "Value": value}]
        elif(data_type == 3):
            if(user_id in users_bmi):
                users_bmi[user_id].append({"Date": date, "Value": value})
            else: 
                users_bmi[user_id] = [{"Date": date, "Value": value}]
    
    #print("Users steps: ", users_steps)
    #print("Users bmi: ", users_bmi)
    return users_steps, users_bmi
    

    
#
#Funzione per costruire gli stati e le azioni dell'environment
#

def buildEnv(dataset = training_collection, bmi_binning = False, cleaning = "cut", reward_policy = "unit"):
    P = {}
    nS = num_bins_steps**days_interval
    nA = num_bins_steps

    P = {s : {a : [(1.0 , nextState(s,a), 0, False)] for a in range(nA)} for s in range(nS)}
    
    dates = getDates()

    users_steps, users_bmi = retrieve(dataset, cleaning)

    users_count = 0 

    for user in users_bmi:
        users_count += 1
        #if(users_count == max_users):
            #break
        
        if(user not in users_steps):
            continue
            
        if(len(users_bmi[user]) < num_bins_bmi or len(users_steps[user]) < num_bins_steps ):
            continue
        
        #print("----------User ", user, "-----------")
        starting_window_index = days_interval
        
        steps_values = users_steps[user]
        if(cleaning == "cut"):
            steps_values = cutTails(steps_values)
        steps_bins_values = getBins(mergeSortByDate(steps_values), num_bins_steps)
        
        bmi_values = mergeSortByDate(users_bmi[user])
        if(bmi_binning == True):
            bmi_values = getBins(bmi_values, num_bins_bmi, quartile = True)
        #print("Bins degli steps: ", steps_bins_values)
        #print("\nBins dei bmi: ", bmi_bins_values)
        finished = False
        
        while not finished:
            window_dates_indices = getValidIndices(bmi_values, starting_window_index, dates)
            if(window_dates_indices == -1):
                #print("Utente processato")
                finished = True
            else:
                #print(window_dates_indices)
                starting_date_index = window_dates_indices[0]
                starting_date = dates[starting_date_index]
                
                ending_date_index = window_dates_indices[1]
                ending_date = dates[ending_date_index]
                
                old_bmi = getValueByDate(bmi_values, starting_date) 
                future_bmi = getValueByDate(bmi_values, ending_date)
                #print("Old BMI: ", old_bmi, " Future BMI: ", future_bmi)
                
                if(bmi_binning == True):
                    reward = (old_bmi - future_bmi)/2
                    #print("\nReward ", reward)
                else:
                    delta = old_bmi - future_bmi
                    if((100*abs(delta))/old_bmi < min_bmi_variation_percentage):
                        #print("Non c'è variazione superiore al ", min_bmi_variation_percentage, "%")
                        reward = 0
                    else:
                        #print("Variazione significativa")
                        if(reward_policy == "delta"):
                            reward = delta
                        elif(reward_policy == "unit"):
                            if(delta >0):
                                reward = 1
                            elif(delta<0):
                                reward = -1

                #print("Reward: ", reward)
                
                if(reward != 0):
                    starting_index = starting_date_index - (days_interval)
                    ending_index = starting_date_index
                    
                    tmp = {}
                    actions_counter = 0
                    while ending_index <= ending_date_index:
                        interval = dates[starting_index:ending_index+1]
                        weekly_steps = getvaluesByDateInterval(steps_bins_values, interval)
                        if(weekly_steps == -1):
                            starting_index += 1
                            ending_index += 1
                            continue
                        else:
                            state = encode(weekly_steps[:days_interval])
                            action = weekly_steps[days_interval:][0]
                            nextstate = encode(weekly_steps[1:])
                            actions_counter += 1
                            #print("State ", decode(state), " Action: ", action)
                            if(state in tmp):
                                if(action not in tmp[state]):
                                    tmp[state][action] = [(1.0, nextstate, reward, False)]
                                else:
                                    prev_reward = tmp[state][action][0][2]
                                    tmp[state][action] = [(1.0, nextstate, prev_reward + reward, False)]
                    
                            else:
                                tmp[state] = {action : [(1.0, nextstate, reward, False)]}
                    
                    
                            starting_index += 1
                            ending_index += 1
                    
                    #print("Actions_counter: ", actions_counter)
                    if(actions_counter/(ending_date_index-starting_date_index+1) > weekly_actions_threshold):
                        #print("Actions counter superiore alla soglia")
                        for state in tmp:
                            for action in tmp[state]:
                                prev_reward = P[state][action][0][2]
                                tmp_reward = tmp[state][action][0][2]
                                nextstate = P[state][action][0][1]
                                P[state][action] = [(1.0, nextstate, prev_reward + tmp_reward, False)]
                
                starting_window_index = ending_date_index
        
        
    '''
    for s in P:
        actions = P[s]
        print("\n\nStato: ", decode(s), ":")
        for a in actions:
            p, n_s, r, d = P[s][a][0]
            
            if(r > max_reward):
                P[s][a] = [(p, n_s, max_reward, d)]
            elif(r < min_reward):
                P[s][a] = [(p, n_s, min_reward, d)]
            
            print("       Action: ", a, ", Probability: ", p , ", Next state: ", decode(n_s), ", Reward: ", P[s][a][0][2])
    '''
    print(users_count)
    return P


# In[5]:


Ptest = buildEnv()


# In[3]:


#
#Funzione che memorizza l'insieme degli stati-azioni in mongo
#
def storeEnvironment():
    P = buildEnv()
    for state in P:
        for action in P[state]:
            p, n_s, reward, done = P[state][action][0]
            doc = {"State": int(state),
                   "Action": int(action),
                   "Probability": float(p),
                   "Next_state": int(n_s),
                   "Reward": float(reward),
                   "Done": done
                    }
            environment_collection.insert_one(doc)


# In[25]:


storeEnvironment()


# In[4]:


#
#L'environment vero e proprio
#
class HRSEnv(discrete.DiscreteEnv):
    
    def __init__(self):
        P = {}
        nS = num_bins_steps**days_interval
        nA = num_bins_steps
        isd = np.zeros(nS)
        
        cursor = environment_collection.find()
        for doc in cursor:
            state = doc["State"]
            action = doc["Action"]
            prob = doc["Probability"]
            next_state = doc["Next_state"]
            reward = doc["Reward"]
            done = doc["Done"]
            if(state in P):
                P[state][action] = [(prob, next_state, reward, done)]
                
            else:
                P[state] = {action: [(prob, next_state, reward, done)]}
                isd[state] += 1
            
        '''
        for s in P:
            ac = P[s]
            print("\n\nStato: ", decode(s), ":")
            for a in ac:
                p, nst, re, d = P[s][a][0]
                print("       Action: ", a, ", Next state: ", decode(nst), ", Reward: ", re)

        '''
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
    
class HRSEnvTest(discrete.DiscreteEnv):
    
    def __init__(self):
        P = buildEnv()
        #P = Ptest
        nS = num_bins_steps**days_interval
        nA = num_bins_steps
        isd = np.zeros(nS)
            
        
        for s in P:
            ac = P[s]
            isd[s] += 1
            print("\n\nStato: ", decode(s), ":")
        
            for a in ac:
                p, nst, re, d = P[s][a][0]
                print("       Action: ", a, ", Next state: ", decode(nst), ", Reward: ", re)
    
        
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)


# In[6]:


envTest = HRSEnv()
#envTest = HRSEnvTest()


# In[7]:


#Algoritmo di q_learning
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, num_explorations , discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in range(num_explorations):
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            #if done:
                #break
                
            state = next_state
    
    return Q


#Altro algoritmo di q_learning
def QL(env, num_episodes, num_explorations, alpha = 0.1, gamma = 0.6, epsilon = 0.2):

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # For plotting metrics
    #all_epochs = []
    #all_penalties = []

    #state = env.reset()
    for i in range(1, num_episodes):
        state = env.reset()
        for j in range(1, num_explorations):

            #epochs, penalties, reward, = 0, 0, 0
            #done = False

            #while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            #if reward == -10:
                #penalties += 1

            state = next_state
            #epochs += 1

        if i % 100 == 0:
            #clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    
    '''
    for s in range(len(q_table)):
        print("\nState:", decode(s))
        print("Actions: ")
        for a in range(len(q_table[s])):
            print("       ", a, ": ", q_table[s][a] )
    '''        
            
    return q_table


# In[46]:


q_table = QL(envTest, 2000, 100000, alpha = 0.6, gamma = 0.6, epsilon = 0.2)
#q_table = q_learning(envTest, 1000, 50000, discount_factor = 0.6, alpha = 0.2, epsilon = 0.2)


# In[8]:


def policyMaker(dataset = training_collection, policy = "popular", cleaning= "cut"):
    P = {}
    table = np.zeros([num_bins_steps**days_interval, num_bins_steps])
    
    dates = getDates()
    
    users_steps, users_bmi = retrieve(dataset, cleaning)
    
    users_count = 0 
    
    if(policy == "popular"):
        table = most_popular_actions(users_steps, dates, table, cleaning)
    elif(policy == "statistical"):
        table = statistically_better_actions(users_steps, users_bmi, dates, table, cleaning)
    
    return table
    
    
    
def most_popular_actions(users_steps, dates, table, cleaning):
    for user in users_steps:
        starting_index = 0
        ending_index = days_interval
        
        steps_values = users_steps[user]
        #print("Passi: ", steps_values)
        if(cleaning == "cut"):
            steps_values = cutTails(steps_values)
        steps_bins_values = getBins(mergeSortByDate(steps_values), num_bins_steps)
        
        while ending_index < len(dates):
            interval = dates[starting_index:ending_index+1]
            #print("Intervallo: ", interval)
            weekly_steps = getvaluesByDateInterval(steps_bins_values, interval)
            if(weekly_steps == -1):
                starting_index += 1
                ending_index += 1
                continue
            else:
                state = encode(weekly_steps[:days_interval])
                action = weekly_steps[days_interval:][0]
                #print("State: ", decode(state), ", Action: ", action)
                table[state, action] += 1
                starting_index += 1
                ending_index += 1
    
    print(table)
    return table
                

    
                
def statistically_better_actions(users_steps, users_bmi, dates, table, cleaning, bmi_binning = False):
    for user in users_bmi:
        
        if(user not in users_steps):
            continue
            
        if(len(users_bmi[user]) < num_bins_bmi or len(users_steps[user]) < num_bins_steps ):
            continue
        starting_window_index = days_interval
        steps_values = users_steps[user]
        if(cleaning == "cut"):
            steps_values = cutTails(steps_values)
        steps_bins_values = getBins(mergeSortByDate(steps_values), num_bins_steps)
        bmi_values = mergeSortByDate(users_bmi[user])
        if(bmi_binning == True):
            bmi_values = getBins(bmi_values, num_bins_bmi)
        #print("Bins degli steps: ", steps_bins_values)
        #print("\nBins dei bmi: ", bmi_bins_values)
        finished = False
        
        while not finished:
            window_dates_indices = getValidIndices(bmi_values, starting_window_index, dates)
            if(window_dates_indices == -1):
                print("Finished a user")
                finished = True
            else:
                print(window_dates_indices)
                starting_date_index = window_dates_indices[0]
                starting_date = dates[starting_date_index]
                ending_date_index = window_dates_indices[1]
                ending_date = dates[ending_date_index]
                old_bmi = getValueByDate(bmi_values, starting_date) 
                future_bmi = getValueByDate(bmi_values, ending_date)
                
                if(bmi_binning == True):
                    weight_loss = future_bmi < old_bmi
                    #print("\nReward ", reward)
                else:
                    delta = old_bmi - future_bmi
                    if((100*abs(delta))/old_bmi < min_bmi_variation_percentage):
                        weight_loss = False 
                    else:
                        weight_loss = delta > 0
                print("Weight Loss: ", weight_loss)
                if(weight_loss):
                
                    starting_index = starting_date_index - (days_interval)
                    ending_index = starting_date_index
                    
                    tmp = {}
                    actions_counter = 0
                    while ending_index <= ending_date_index:
                        interval = dates[starting_index:ending_index+1]
                        weekly_steps = getvaluesByDateInterval(steps_bins_values, interval)
                        if(weekly_steps == -1):
                            starting_index += 1
                            ending_index += 1
                            continue
                        else:
                            actions_counter += 1
                            state = encode(weekly_steps[:days_interval])
                            print("State: ", decode(state))
                            action = weekly_steps[days_interval:][0]
                            print("Action: ", action)
                            if(state in tmp):
                                if(action not in tmp[state]):
                                    tmp[state][action] = 1
                                else:
                                    
                                    tmp[state][action] += 1
                                    
                            else:
                                tmp[state] = {action : 1}
                    
                            starting_index += 1
                            ending_index += 1
                    print("Actions counter: ", actions_counter)
                    if(actions_counter/(ending_date_index-starting_date_index+1) > weekly_actions_threshold):
                        print("Actions counter superiore alla soglia")
                        for state in tmp:
                            for action in tmp[state]:
                                table[state, action] += tmp[state][action]
                
                
                starting_window_index = ending_date_index
    
    print(table)
    return table



def evaluatePolicy(dataset = test_collection, policy = "random", table = None, bmi_binning = False, cleaning = "cut"):
    weight_loss_matching, weight_gain_matching, weight_loss_not_matching, weight_gain_not_matching = 0, 0, 0, 0
    
    dates = getDates()
    
    users_steps, users_bmi = retrieve(dataset, cleaning)
    
    users_count = 0
        
    for user in users_bmi:
        if(user not in users_steps):
            continue
            
        if(len(users_bmi[user]) < num_bins_bmi or len(users_steps[user]) < num_bins_steps ):
            continue
        #print("-------User ", user, "--------")    
        starting_window_index = days_interval
        steps_values = users_steps[user]
        if(cleaning == "cut"):
            steps_values = cutTails(steps_values)
        steps_bins_values = getBins(mergeSortByDate(steps_values), num_bins_steps)
        bmi_values = mergeSortByDate(users_bmi[user])
        if(bmi_binning == True):
            bmi_values = getBins(bmi_values, num_bins_bmi)
        
        finished = False

        while not finished:
            window_dates_indices = getValidIndices(bmi_values, starting_window_index, dates)
            if(window_dates_indices == -1):
                #print("Utente processato")
                finished = True
            else:
                #print(window_dates_indices)
                starting_date_index = window_dates_indices[0]
                starting_date = dates[starting_date_index]
                ending_date_index = window_dates_indices[1]
                ending_date = dates[ending_date_index]
                old_bmi = getValueByDate(bmi_values, starting_date) 
                future_bmi = getValueByDate(bmi_values, ending_date)
                
                if(bmi_binning == True):
                    weight_loss = old_bmi -future_bmi
                    #print("\nReward ", reward)
                else:
                    delta = old_bmi - future_bmi
                    if((100*abs(delta))/old_bmi < min_bmi_variation_percentage):
                        #print("Non c'è variazione superiore al ", min_bmi_variation_percentage, "%")
                        weight_loss = 0 
                    else:
                        #print("Variazione significativa")
                        weight_loss = delta
                
                #print("Weight Loss: ", weight_loss)
                if(weight_loss != 0):
                    starting_index = starting_date_index - (days_interval)
                    ending_index = starting_date_index

                    actions_counter, followed_recommendations = 0, 0

                    while ending_index <= ending_date_index:
                        interval = dates[starting_index:ending_index+1]
                        weekly_steps = getvaluesByDateInterval(steps_bins_values, interval)
                        if(weekly_steps == -1):
                            starting_index += 1
                            ending_index += 1
                            continue
                        else:
                            state = encode(weekly_steps[:days_interval])
                            #print("State: ", decode(state))
                            action = weekly_steps[days_interval:][0]
                            #print("Action: ", action)
                            nextstate = encode(weekly_steps[1:])
                            actions_counter +=1

                            #More Steps case
                            if(policy == "more_steps"):

                                if(weekly_steps[days_interval-1] < num_bins_steps-1):
                                    #suggested_actions = list(range(weekly_steps[days_interval-1] +1, num_bins_steps))
                                    suggested_actions = [weekly_steps[days_interval-1] +1]
                                else:
                                    suggested_actions = [num_bins_steps-1]

                            #Random Case
                            elif(policy == "random"):
                                suggested_actions = [random.randint(0,num_bins_steps-1)]

                            #Q_learning, Statistical, or Popular case
                            elif(policy == "q_learning" or policy == "statistical" or policy == "popular"):
                                suggested_actions = [np.argmax(table[state])]



                            #print("Suggested Action: ", suggested_actions)
                            if(action in suggested_actions):
                                #print("Followed Recommendation")
                                followed_recommendations += 1 


                            starting_index += 1
                            ending_index += 1

                    #print("Actions Counter: ", actions_counter)
                    if(actions_counter/(ending_date_index-starting_date_index+1) > weekly_actions_threshold):
                        #print("Actions Counter superiora alla soglia")
                        if(weight_loss > 0):
                            #print("Perdita di peso, setting di wlm e wlnm")
                            weight_loss_matching += followed_recommendations
                            weight_loss_not_matching += (actions_counter-followed_recommendations)
                            #print("Weight loss matching: ", weight_loss_matching)
                            #print("Weight loss not matching", weight_loss_not_matching)
                        elif(weight_loss < 0):
                            #print("Guadagno di peso, detting di wgm e wgnm")
                            weight_gain_matching += followed_recommendations
                            weight_gain_not_matching += (actions_counter-followed_recommendations)
                            #print("Weight gain matching: ", weight_gain_matching)
                            #print("Weight gain not matching: ", weight_gain_not_matching)
                
                starting_window_index = ending_date_index

        users_count += 1
    
    #print("Totatle utenti: ", users_count)
    precision = weight_loss_matching/(weight_loss_matching + weight_loss_not_matching)
    accuracy = (weight_gain_matching + weight_loss_matching)/(weight_gain_matching + weight_loss_matching + weight_loss_not_matching + weight_gain_not_matching)
    #print(policy)
    #print("Precision: ", precision, "\nAccuracy: ", accuracy)
    return precision, accuracy


# In[48]:


#evaluatePolicy()
#evaluatePolicy(policy = "more_steps")
#evaluatePolicy(policy = "popular", table = policyMaker(policy = "popular"))
evaluatePolicy(policy = "statistical", table = policyMaker(policy = "statistical"))
#evaluatePolicy(policy = "q_learning", table = q_table)


# In[ ]:


alpha_values = [0.5, 0.6, 0.7]
gamma_values = [0.5, 0.6, 0.7]
epsilon_values = [0.2, 0.3]

for a in alpha_values:
    for g in gamma_values:
        for e in epsilon_values:
            q_table = QL(envTest, 2000, 100000, alpha = a, gamma = g, epsilon = e)
            prec, acc = evaluatePolicy(policy = "q_learning", table = q_table)
            print("For Alpha = ", a, " Gamma = ", g, " ed Epsilon = ", e)
            print("Precision = ", prec)
            print("Accuracy = ", acc)

