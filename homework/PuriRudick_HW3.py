# -*- coding: utf-8 -*-
# @Author: Puri Rudick
# @Date:   2022-03-02 20:49:04
# @Last Modified by:   Your name
# @Last Modified time: 2022-03-05 22:29:11


# Decision making with Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch.
# You have to pick a restaurant that’s best for everyone.
# Then you should decided if you should split into two groups so everyone is happier. 
 
# Despite the simplicity of the process you will need to make decisions regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning. 

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%% 
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
people = {'Jane': {'willingness to travel': 4.5,
				  'desire for new experience': 3.0,
				  'cost': 1.0,
				#   'Mexican food': 3.5,
				#   'hipster points': 2.0,
				  'vegetarian': 0.2,
				  },
		  'Casey': {'willingness to travel': 0.5,
				  'desire for new experience': 4.2,
				  'cost': 2.5,
				#   'Mexican food': 4.5,
				#   'hipster points': 1.0,
				  'vegetarian': 3.5,
				  },
		  'Rylee': {'willingness to travel': 3.5 ,
				  'desire for new experience': 2.0,
				  'cost': 4.5,
				#   'Mexican food': 3.0,
				#   'hipster points': 2.0,
				  'vegetarian': 0.1,
				  },
		  'Pepper': {'willingness to travel': 4.0,
				  'desire for new experience': 4.0,
				  'cost': 4.0,
				#   'Mexican food': 3.0,
				#   'hipster points': 2.5,
				  'vegetarian': 0.0,
				  },
		  'Dale': {'willingness to travel': 2.0,
				  'desire for new experience': 2.5,
				  'cost': 3.5,
				#   'Mexican food': 3.0,
				#   'hipster points': 1.0,
				  'vegetarian': 4.5,
				  },
		  'Lee': {'willingness to travel': 2.0,
				  'desire for new experience': 2.0,
				  'cost': 2.5,
				#   'Mexican food': 2.5,
				#   'hipster points': 1.7,
				  'vegetarian': 0.2,
				  },
		  'Molly': {'willingness to travel': 2.1,
				  'desire for new experience': 1.5,
				  'cost': 4.7,
				#   'Mexican food': 3.2,
				#   'hipster points': 4.1,
				  'vegetarian': 0.2,
				  },
		  'Eric': {'willingness to travel': 4.5,
				  'desire for new experience': 0.2,
				  'cost': 3.8,
				#   'Mexican food': 3.0,
				#   'hipster points': 2.5,
				  'vegetarian': 0.1,
				  },
		  'Reed': {'willingness to travel': 3.0,
				  'desire for new experience': 4.1,
				  'cost': 5.0,
				#   'Mexican food': 3.1,
				#   'hipster points': 0.5,
				  'vegetarian': 0.1,
				  },
		  'Nicky': {'willingness to travel': 3.1,
				  'desire for new experience': 2.1,	
				  'cost': 5.0,
				#   'Mexican food': 2.1,
				#   'hipster points': 1.1,
				  'vegetarian': 1.5,
				  }                  
		  }


#%%
# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
M_people = []
people_list = []	# To track row labels
survey_list = []	# To track column labels

for i in people:
	M = []
	people_list.append(i)
	s = []
	for j in people[i]:
		M.append(people[i][j])
		s.append(j)
	M_people.append(M)
	survey_list.append(s)

M_people = np.array(M_people)
print(M_people)
print(people_list)

survey_list = set(tuple(i) for i in survey_list)
survey_list =[item for t in survey_list for item in t]
print(survey_list)

# Print the matrix as dataframe with column and row namesß
M_people_df = pd.DataFrame(M_people, columns=survey_list, index=people_list)
M_people_df


#%%
# Next you collected data from an internet website. You got the following information.
restaurants  = {'Chick-fil-A':{'distance': 4,
						'novelty': 0.7,
						'cost': 4.5,
						# 'average rating': 4.5,
						# 'cuisine': 3,
						'vegetarian': 1.5
						},
			  	 'El Malecon':{'distance': 5,
						'novelty': 4.7,
						'cost': 3,
						# 'average rating': 4.1,
						# 'cuisine': 4.7,
						'vegetarian': 2
					  },
			  	 'Thai House':{'distance': 1,
						'novelty': 5,
						'cost': 1.5,
						# 'average rating': 4.5,
						# 'cuisine': 4.8,
						'vegetarian': 3
					  },                      
			  	 'Librado':{'distance': 4.2,
						'novelty': 4.5,
						'cost': 1,
						# 'average rating': 4.8,
						# 'cuisine': 4.2,
						'vegetarian': 4.5
					  },
			  	 'The Tailgate':{'distance': 4.5,
						'novelty': 1.7,
						'cost': 4,
						# 'average rating': 4.0,
						# 'cuisine': 3,
						'vegetarian': 0.5
					  },
			  	 'Grub Burger Bar':{'distance': 3.5,
						'novelty': 2.5,
						'cost': 2.2,
						# 'average rating': 4.9,
						# 'cuisine': 3.7,
						'vegetarian': 2.2
					  },
			  	 'Texas Roadhouse':{'distance': 4.1,
						'novelty': 4,
						'cost': 2.1,
						# 'average rating': 4.5,
						# 'cuisine': 4.0,
						'vegetarian': 2.1
					  },
			  	 'Clear Spring':{'distance': 4.2,
						'novelty': 2.7,
						'cost': 4.1,
						# 'average rating': 4.6,
						# 'cuisine': 4.2,
						'vegetarian': 4.5
					  },
			  	 'La Mision':{'distance': 4.4,
						'novelty': 3.8,
						'cost': 3.4,
						# 'average rating': 4.1,
						# 'cuisine': 4.7,
						'vegetarian': 2.1
					  },
			  	 'Osaka':{'distance': 4.4,
						'novelty': 4.1,
						'cost': 4.0,
						# 'average rating': 4.2,
						# 'cuisine': 3.2,
						'vegetarian': 3.8
					  }                      
}


#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
M_restaurants = []
restaurants_list = []	# To track row labels
rating_list = []	# To track column labels

for i in restaurants:
	M = []
	restaurants_list.append(i)
	s = []
	for j in restaurants[i]:
		M.append(restaurants[i][j])
		s.append(j)
	M_restaurants.append(M)
	rating_list.append(s)

M_restaurants = np.array(M_restaurants)
print(M_restaurants)
print(restaurants_list)

rating_list = set(tuple(i) for i in rating_list)
rating_list =[item for t in rating_list for item in t]
print(rating_list)

# Print the matrix as dataframe with column and row namesß
M_restaurants_df = pd.DataFrame(M_restaurants, columns=rating_list, index=restaurants_list)
M_restaurants_df


# %%
# Choose a person and compute(using a linear combination) the top restaurant for them.
# What does each entry in the resulting vector represent? 

# The person I choose is 'Casey'
casey_pick = np.dot(M_people[people_list.index('Casey')], M_restaurants.T)
casey_first_pick = np.argmax(casey_pick)

print(f'Casey''s top restauratn is ', restaurants_list[casey_first_pick])


#%%
# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
M_usr_x_rest=np.dot(M_people,M_restaurants.T)
print(M_usr_x_rest)

# What does the a_ij matrix represent?
print('Row labels are people: ', people_list)
print('Column labels are restaurant: ', restaurants_list)


# %%
# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?
restaurant_total_score = M_usr_x_rest.sum(axis=0) 

s = 0
print('Restaurant over all score (from highest to lowest):')
for i in reversed(np.argsort(restaurant_total_score)):
    s += 1
    print('#%d  %s - Total Score: %s' % (s, restaurants_list[i], '{0:.2f}'.format(restaurant_total_score[i])))


# %%
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.
# Do the same as above to generate the optimal restaurant choice.
sidx = np.argsort(M_usr_x_rest, axis=1)

m,n = M_usr_x_rest.shape

# Initialize output array
M_usr_x_rest_rank = np.empty((m,n),dtype=int)
M_usr_x_rest_rank[np.arange(m)[:,None], sidx] = np.arange(n)	# Rank 9 is the highest score, while rank 0 is the lowest score 

# Sum all columns in M_usr_x_rest_rank to get the optimal restaurant for all users.
restaurant_total_rank_score = M_usr_x_rest_rank.sum(axis=0)

s = 0
print('Restaurant ranking score (from highest to lowest):')
for i in reversed(np.argsort(restaurant_total_rank_score)):
    s += 1
    print('#%d  %s - Total Rank Score: %s' % (s, restaurants_list[i], '{0:.2f}'.format(restaurant_total_rank_score[i])))


# %%
# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?
'''
The different between the two happens because the first one uses raw score with weights from the survey. 
While the second one is only based on the total of ranking number.
In reality, this represents tthe important of features' weight.  Which most of the time are not equaly important.
'''

# How should you preprocess your data to remove this problem. 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?

# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 

# %%
