#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Class-Choice Problems:

# Suppose that we have a set I of students and a set S of classes.
# Each student has a preference (a linear order) over S' (S and the empty set). Each class has a preference over I.
# Each student has a maximum number of classes they can take. Each class has a maximum number of students.
# What is the optimal way to sort students into classes?

# See this term paper I wrote in 2021: https://github.com/camzgray/hmc-algorithm/blob/main/Ec_117_Term_Paper.pdf

# This project came from a real-life application in 2023.
# I had a group of people who were each being assigned to a cards, and had preferences over the cards.
# Each person could get to assigned to one slot, and each card had one or two slots.
# Only the top few "acceptable" cards for each person were available. I used two strategies to solve this problem:
# 1) Minimizing a cost function as in the 2021 paper.
# 2) Finding an algorithm that returned an acceptable or near-acceptable matching (where each person got one of their acceptable cards).

import copy
import csv
import numpy as np
# Uses CVXPY: https://www.cvxpy.org/install/index.html
import cvxpy as cp
import cvxopt as cpt


# In[190]:


# Data:

# Our CSV file is in the format where each row represents a card, and in each column of the row is a person with preference
# For example, for the "Card 0" row, we may have "Abagail (1),Charlie (3),,," where we may have empty columns

#MYFILE = ""
#file = open(MYFILE,"r")
#data = list(csv.reader(file, delimiter=","))
#file.close()

# Example data
data = [["Abagail (1)","Bill (2)"],["Bill (1)"],["Charlie (2)"],["Abagail (2)","Charlie (1)","Danielle (1)"]]

CARD_NAMES = False # If the first column of the file is names of cards
cardList = range(len(data))
if CARD_NAMES:
    cardNames = []
    for i in cardList:
        cardNames.append(data[i][0])
        data[i].pop(0)
else:
    cardNames = cardList

# Removes empty entries
for j in range(len(data[0])):
    for i in data:
        try:
            i.remove('')
        except ValueError:
            pass

# Splits each entry into the name and the order of their preference
for i in range(len(data)):
    for j in range(len(data[i])):
        split_text = data[i][j].split()
        data[i][j] = (split_text[0],int(split_text[1][1]))

personList = []
for i in data:
    for j in i:
        if not(j[0] in personList):
            personList.append(j[0])
personList.sort()

CARD_MAXIMUMS = [1,1,2,1] # The maximum number of people that can be assigned to each card

if len(CARD_MAXIMUMS) != len(cardList):
    print("Warning! The number of cards does not equal the length of the maximum number of people that can get assigned a card.")
    print("You may need to update CARD_MAXIMUMS.")

ALLOW_UNASSIGNED = True # If we allow someone to be assigned to no card

if ALLOW_UNASSIGNED == False and len(personList) > sum(CARD_MAXIMUMS):
    print("Warning! There are more people than there are slots, and people not being assigned is not allowed.")

REQUIRE_FILLING = True # If true, we require that all cards are filled before any allowing two people in one card

ACCEPTABLE_LEVEL = 2 # This is the maximum ranking that we consider acceptable

preferenceMatrix = []
for card in data:
    new_card = []
    for person in card:
        if person[1] <= ACCEPTABLE_LEVEL:
            new_card.append(person[0])
    preferenceMatrix.append(new_card)
            
pairings = [] # This will be a list of pairings of people with cards


# In[194]:


# 2) An algorithm to find acceptable matchings. It won't work in all cases.
# TODO: This currently does not factor in that some cards might have multiple slots.
# TODO: This algorithm takes too long for large amounts of preferences.

def make_decisions_1(pMatrix,pairings):
    """
    From a preference matrix and a list of pairings, takes any card with only one acceptable person and pairs
    them with that card. Returns a new preference matrix (not including the paired person) and list of pairings in a tuple.
    """
    newPairings = copy.deepcopy(pairings)
    pMatrixNew = copy.deepcopy(pMatrix)
    for i in range(len(pMatrixNew)):
        if len(pMatrixNew[i]) == 1:
            match = pMatrixNew[i][0]
            newPairings.append((i,match))
            for j in range(len(pMatrixNew)):
                tempList = pMatrixNew[j]
                if match in tempList:
                    tempList.remove(match)
                    pMatrixNew[j] = tempList
    return (pMatrixNew,newPairings)

def make_decisions_1_rec(pMatrix,pairings):
    """
    Recursively performs make_decisions_1 until all cards with a single person have been paired.
    """
    results = make_decisions_1(pMatrix,pairings)
    results2 = make_decisions_1(results[0],results[1])
    if results == results2:
        return results2
    else:
        return make_decisions_1_rec(results2[0],results2[1])
    
def split_tree(pMatrixList,pairingList):
    """
    Takes in a list of preference matrices and a list of lists of pairings. For each card with two acceptable pairings,
        adds each possible new preference matrix and pairing to the list, choosing a different pairing for each.
    """
    pMatrixListNew = copy.deepcopy(pMatrixList)
    pairingListNew = copy.deepcopy(pairingList)
    for q in range(len(pMatrixListNew)):
        for i in range(len(pMatrixListNew[q])):
            if len(pMatrixListNew[q][i]) == 2:
                # store current result
                stored_pMatrix = copy.deepcopy(pMatrixListNew[q])
                stored_pairing = copy.deepcopy(pairingListNew[q])
                res1 = pMatrixListNew[q][i][0]
                res2 = pMatrixListNew[q][i][1]
                # change the first result
                pMatrixListNew[q][i] = [res1]
                pairingListNew[q].append((i,res1))
                # clean first result 
                for j in range(len(pMatrixListNew[q])):
                    tempList = pMatrixListNew[q][j]
                    if res1 in tempList:
                        tempList.remove(res1)
                        pMatrixListNew[q][j] = tempList
                # add a new splitting with the second result
                pMatrixListNew.append(stored_pMatrix)
                pairingListNew.append(stored_pairing)
                pMatrixListNew[-1][i] = [res2]
                pairingListNew[-1].append((i,res2))
                # clean second result
                for j in range(len(pMatrixListNew[-1])):
                    tempList = pMatrixListNew[-1][j]
                    if res2 in tempList:
                        tempList.remove(res2)
                        pMatrixListNew[-1][j] = tempList
    return (pMatrixListNew,pairingListNew)

def split_decisions(pMatrixList,pairingList):
    """
    Recursively performs split_tree, then performing make_decisions_1_rec on each option until all cards with
        one or two people have been placed.
    """
    results = split_tree(pMatrixList,pairingList)
    for q in range(len(results[0])):
        results[0][q],results[1][q] = make_decisions_1_rec(results[0][q],results[1][q])
    if results == (pMatrixList,pairingList):
        return results
    else:
        return split_decisions(results[0],results[1])

def pairing_stats(cList,pList,pMatrix,pairings):
    """
    Given a list of cards, a list of people, a preference matrix, and a list of pairings, returns statistics on the pairing.
    """
    print("Pairings: ",pairings)
    print("Number of Unpaired Cards: ",len(cList) - len(pairings))
    print("Number of Unpaired People: ",len(pList) - len(pairings))
    unhappyPeople = 0
    unhappyPairs = []
    for pair in pairings:
        if not(pair[1] in pMatrix[pair[0]]):
            unhappyPeople += 1
            unhappyPairs.append(pair)
    print("People unhappy with their pairings: ",unhappyPeople)
    print("Unhappy pairings: ",unhappyPairs)


# In[203]:


nres = split_decisions([preferenceMatrix],[[]])


# In[204]:


goodPairs = []
for q in range(len(nres[0])):
    pL = len(personList)
    if len(nres[1][q]) == pL: # all people are paired
        tempPair = []
        for i in range(pL):
            for j in range(pL):
                if nres[1][q][j][0] == i:
                    tempPair.append(nres[1][q][j])
        if not(tempPair in goodPairs):
            goodPairs.append(tempPair)

for i in range(len(goodPairs)):
    pairing_stats(cardList,personList,preferenceMatrix,goodPairs[i])


# In[208]:


# 1) Integer Linear Programming:
# This is taken and modified from the my 2021 paper.

NUM_OF_STUDENTS = len(personList)
NUM_OF_CLASSES = len(cardList)
MAX_CLASSES = 1 # Maximum number of classes a student can have


#NUM_OF_STUDENTS = 3
#NUM_OF_CLASSES = 3

# Initializes the variables - there must be one for each class that a student can be assigned to
# In our case each person only gets one card, so we only need a single variable
# TODO: Make it so that changing MAX_CLASSES automatically creates new variables
# NUM_OF_CLASSES + 1 as a person/student can be assigned to no card/class at all
x1 = cp.Variable((NUM_OF_STUDENTS,(NUM_OF_CLASSES+1)),integer=True)
# x2 = cp.Variable((NUM_OF_STUDENTS,(NUM_OF_CLASSES+1)),integer=True)

preferences = np.full([NUM_OF_STUDENTS, NUM_OF_CLASSES+1],-1, dtype=int)
# Fills out the preference matrix, where each row is a student, and the entries are their card preferences listed in order
# 0 is the first card, and -1 is unassigned
# Unlisted preferences are considered unassigned
for i in range(len(data)):
    for j in range(len(data[i])):
        value = data[i][j]
        preferences[personList.index(value[0])][value[1]-1] = i
# This is the priority that the classes have over students (assuming all have the same priority)
priority = np.array(range(1,NUM_OF_STUDENTS+1))
# The maximum number of students for each class (including the empty class)
if ALLOW_UNASSIGNED:
    max_students = np.array(CARD_MAXIMUMS + [MAX_CLASSES*NUM_OF_STUDENTS])
else:
    max_students = np.array(CARD_MAXIMUMS + [0])
# The minimum number of students for each class
if REQUIRE_FILLING and NUM_OF_STUDENTS >= NUM_OF_CLASSES:
    # If we require filling all cards (and have enough people to do so)
    min_students = np.ones([NUM_OF_CLASSES+1],dtype=int)
    min_students[-1] = 0 # Don't need to fill unassigned
else:
    min_students = np.zeros([NUM_OF_CLASSES+1],dtype=int)
    
# Constants determine the relative weight of preference to priority.
# In our case, priority is only used as a tie-breaker.
PREFERENCE_MULTIPLIER = 1 # C_1
PRIORITY_MULTIPLIER = 0 # C_2

# TODO: planning to hard-code in costs for a specific pairing - could either create an infinite cost, or make a constraint

UNACCEPTABLE_COST = 0 # The additional cost of getting an unacceptable card (one didn't have a preference for)

def cost_func(i,s,pref,prior):
    """Takes in the student, class and their
    preferences and priorities, returns cost"""
    this_pref = pref[i-1]
    empty_loc = np.argwhere(this_pref==-1).flatten()[0] # The cost of getting an unlisted card or unassigned is one's final ranking.
    if s == NUM_OF_CLASSES:
        return empty_loc
    school_loc_list = np.argwhere(this_pref==s).flatten()
    if len(school_loc_list) == 0:
        school_loc = empty_loc + UNACCEPTABLE_COST
    else:
        school_loc = school_loc_list[0]
    c_pref = school_loc*PREFERENCE_MULTIPLIER
    c_prior = (np.argwhere(prior==i).flatten()[0])*PRIORITY_MULTIPLIER
    # If the priority cost is constant across classes for each person, I don't think it does anything
    return c_pref+c_prior

cost = np.fromfunction(np.vectorize(
    lambda a, b: cost_func(a+1,b,preferences,priority)), 
                       (NUM_OF_STUDENTS,NUM_OF_CLASSES+1) ,dtype=int)

HARD_CODE = False # Do we have hard-coded constraints?
if HARD_CODE:
    print("Warning! You have a hard-coded in penalty.")
HARD_CODE_COST = -10 # What is the penalty?

HARD_CODE_NET = 2*HARD_CODE_COST # the net cost, for computing values

print("Cost Matrix:")
print(cost)
#diag_wzero = np.diagflat(np.ones(NUM_OF_CLASSES+1,dtype=int))
#diag_wzero[0][0] = 0


constraints = [x1>=0,x1<=1,
                cp.sum(x1,axis=1)==1,
                # Must be in at least one class or no class
                cp.sum(x1,axis=0)>=min_students,
                cp.sum(x1,axis=0)<=max_students]
                # Must obey class size limits

obj = cp.Minimize(cp.vec(cost)@cp.vec(x1))

#constraints = [x1>=0,x1<=1,x2>=0,x2<=1,
#               cp.sum(x1,axis=1)==1,cp.sum(x2,axis=1)==1,
#               # Must be in at least one class or no class
#               (x1+x2)@diag_wzero<=1,
#               # Cannot be in the same class twice
#               # but can be in no class twice
#              cp.sum(x1,axis=0) + cp.sum(x2,axis=0)<=max_students]
                # Must obey class size limits
#obj = cp.Minimize(cp.vec(cost)@cp.vec(x1)+cp.vec(cost)@cp.vec(x2))

# Solving the problem
prob = cp.Problem(obj, constraints)
prob.solve(verbose=False)
print("Status:", prob.status)
print("Total Cost:", prob.value-HARD_CODE_NET)


def return_text(values):
    classnum = np.argwhere(values==1)[0][0]
    if classnum == NUM_OF_CLASSES:
        return "No Card"
    else:
        if CARD_NAMES:
            return cardNames[classnum]
        else:
            return "Card " + str(classnum)

def find_ranking(personIndex,data,values):
    classnum = np.argwhere(values==1)[0][0]
    if classnum == NUM_OF_CLASSES:
        return ""
    else:
        for val in data[classnum]:
            if val[0] == personList[personIndex]:
                return "(Ranked " + str(val[1]) + ")"
        return "(Not Ranked)"

for i in range(len(data)):
    for j in range(len(data[i])):
        value = data[i][j]
        preferences[personList.index(value[0])][value[1]-1] = i
print("--------")
print("Assignments by Person:")
for i in range(NUM_OF_STUDENTS):
    print(personList[i],"is assigned to",return_text(x1.value[i]),find_ranking(i,data,x1.value[i]))

values = np.array(x1.value,dtype=int)

def return_text2(valueM):
    personnum = np.argwhere(valueM==1).flatten()
    if len(personnum) == 0:
        return "No one"
    text = personList[personnum[0]]
    if len(personnum)>1:
        for i in range(1,len(personnum)):
            text = text + " and " + personList[personnum[i]]
    return text

print("--------")
print("Assignments by Card:")
for i in range(NUM_OF_CLASSES):
    print(cardNames[i],"is assigned to",return_text2(np.transpose(values)[i]))
print(return_text2(np.transpose(values)[NUM_OF_CLASSES]),"not assigned")
print("--------")
print("Values Matrix:")
print(values)


# In[ ]:




