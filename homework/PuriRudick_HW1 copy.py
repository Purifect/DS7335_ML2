# -*- coding: utf-8 -*-
# @Author: Puri Rudick
# @Date:   2022-01-15 16:26:56
# @Last Modified by:   Your name
# @Last Modified time: 2022-01-18 12:11:54


# Fill in an explanation of each function and an example of how to use it below.


# ************ List ************ #
# List is a collection which is ordered and changeable. Allows duplicate members
mylist = ['apple', 'banana', 'cherry', 'orange']     # create a demo list

# append() – Adds an item to the end of the list.
mylist.append('apple')
print(mylist)

# extend() – Adds the specified list elements (or any iterable) to the end of the current list.
col = ('red', 'yellow', 'green')       # create 2nd demo list
mylist.extend(col)
print(mylist)

# index() – Returns the index of the specified element in the list (first value match).
print(mylist.index('apple'))

# index(value, integer) - The method lets we search the value in the list with starting index. 
# Return 1 when the value is found in the list
# Return ValueError when the value is not found in the list
mylist.index('banana', 1)
# mylist.index('banana', 5)     # return ValueError

# insert(position) – Inserts an element to the list at the specified ‘position’ index.
mylist.insert(5, 'pineapple')
print(mylist)

# remove() – Removes the first matching element from the list.
mylist.remove('apple')
print(mylist)

# pop() – Removes the item at the given index from the list and returns the removed item.
mylist.pop(2)
print(mylist)

# count() – Returns the number of times the specified element appears in the list.
mylist.count('banana')

# reverse() – Reverses the elements of the list.
mylist.reverse()
print(mylist)

# sort() – Sorts the list ascending by default.
mylist.sort()
print(mylist)

# [1]+[1] - Use '+' operator to join the (two) lists and returns a new list. 
print([1]+[1])

# [2]*2 - Use '*' operation to create a new list with the elements repeated the specified number of times.
print([2]*2)

# [1,2][1:] - Slice the list [1,2] from index [1:], which mean from index[1] to the end of the list
print([1,2][1:])
print([1,2,3,4,5][1:])      # slice the list from index [1:]
print([1,2,3,4,5][1:3])      # slice the list from index [1] to index[2]

# [x for x in [2,3]] - Loop through the list
print([x for x in [2,3]])
print([x for x in ['apple', 'banana', 'cherry']])

# [x for x in [1,2] if x == 1] - Loop through the list and only return when the value(s) equals to 1``
print([x for x in [1,2] if x == 1])
print([x for x in [1,2,2,3,3,3] if x == 2])     # loop through the list and only return when the value(s) equals to 1

# [y*2 for x in [[1,2],[3,4]] for y in x] - Loop through 'list of the 2 lists' with x, then multiply that value by 2
print([y*2 for x in [[1,2],[3,4]] for y in x])

# A = [1] - Assigns the list [1] to variabel A
A = [1]
print(A)


# ************ Tuple ************ #
# Tuple is a collection which is ordered and unchangeable. Allows duplicate members.
tp = (('a', 'b'), 'a', 'e', 'i', 'o', 'i', 'u', ('a', 'b'), [3, 4])     # create a demo tuple

# count() – Returns the number of times element appears in the tuple.  The count() method only takes a single argument.
print(tp.count('i'))
print(tp.count(('a', 'b')))    # count tuple element inside tuple
print(tp.count([3, 4]) )   # count list element inside tuple

# index() – Returns the index of the specified element in the tuple (only returns the first occurrence of the matching element.).
# If the element is not found, a ValueError exception is raised.  The syntax of the tuple index() method is: tuple.index(element, start, end)
# when 	element = the element to be searched, start (optional) = start searching from this index, and end (optional) = search the element up to this index
print(tp.index('i'))   # index of an element
#print(tp.index('p'))   # index of an element not present in the tuple, ValueError exception is raised
print(tp.index(('a', 'b'), 4))     # index with start parameter
print(tp.index(('a', 'b'), 0, 5))  # index with start and end parameter

# build a dictionary from tuples – Use the dict() method
tp2 = ((1, 'a'), (2, 'b') ,(3, 'c'), (4, 'd'), (5, 'e'))    # create a demo tuple
tuple_as_dictionary = dict(tp2)
print(tuple_as_dictionary)

# unpack tuples – In packing, we place value into a new tuple while in unpacking we extract those values back into variables.
tp_pack = ('Puri', 'Rudick', 1234)     # tuple packing
(firstname, lastname, id) = tp_pack    # tuple unpacking
print(firstname)
print(lastname)
print(id)



# ************ Dictionary ************ #
# Dictionary is a collection which is ordered** and changeable. No duplicate members.

# a_dict = {'I hate':'you', 'You should':’leave’} – Assigns variable ‘a_dict’ as a dictionary with 2 pairs of key:value.
a_dict = {'I hate':'you', 'You should':'leave'}

# keys() – Returns a view object is ret that displays all the keys. This view object changes according to the changes in the dictionary.
a_dict.keys()   # keys

# items() – Returns a view object that displays a list of dictionary's (key, value) tuple pairs.  Note that The items() method doesn't take any parameters.
a_dict.items()  # items

# hasvalues() - Was originally in earlier versions of python, it is no longer exist in python 3.
# _keys() - Was originally in earlier versions of python, it is no longer exist in python 3.

# ‘never’ in a_dict – Returns True if the value 'never' is in a_dict, returns False if if the value 'never' is NOT in a_dict.
'never' in a_dict   # Returns False since we do not have ‘never’ in a_dict

# del a_dict['me'] – 'del' is used to inplace delete the key that is present in the dictionary.  An exception KeyError is raised if the key is not found and hence non-existence of key has to be handled.
# I commented the code out so it does not raise the KeyError when you run.
# del a_dict['me']    # Returns KeyError: ‘me’, since we do not have ‘me’ as one of the keys in a_dict

# a_dict.clear() – Removes all items from the dictionary a_dict.
a_dict.clear()
print(a_dict)

## dir() - Use dir() to get built in functions
print(dir(dict))


# ************ Sets ************ #
# Set is a collection which is unordered, unchangeable*, and unindexed. No duplicate members.
myset = {'apple', 'banana', 'cherry'}       # create a demo set

# add() – Adds a given element to a set if the element is not present in the set.
myset.add('orange')
print(myset)

# clear() – Removes all elements from the set.
myset.clear()
print(myset)

# copy() – Returns a shallow copy of the set.  The copy() does not take any parameters.
myset = {'apple', 'banana', 'cherry'}
myset2 = myset.copy()   # copy from myset to myset2
print(myset2)

# difference() – Returns the set difference of two sets.
listA = {10, 20, 30, 40, 80}
listB = {100, 30, 80, 40, 60}
print (listA.difference(listB))     # difference A-B
print (listB.difference(listA))     # difference B-A

# discard() – Removes the specified item from the set.
listA.discard(10)
print(listA)

# intersection() – Returns a new set with elements that are common to all sets.
print(listA.intersection(listB))    # intersection

# issubset() – Returns True if all elements of a set are present in another set (passed as an argument).  If not, it returns False.
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}
C = {1, 2, 4, 5}
print(A.issubset(B))    # issubset - Is A subset of B?
print(A.issubset(C))    # issubset - Is A subset of C?

# pop() – Removes a random item from the set and returns removed value.
myset.pop()
print(myset)

# remove() – Removes the specified element from the set.  If the element passed to remove() does not exist, KeyError exception is thrown.
myset = {'apple', 'banana', 'cherry'}
myset.remove('apple')
print(myset)

# union() – Returns a new set with distinct elements from all the sets.  If the argument is not passed to union(), it returns a shallow copy of the set.
# We can also find the union of sets using the | operator.
A = {'a', 'c', 'd'}
B = {'c', 'd', 2 }
C = {1, 2, 3}
A.union(B)      # A union B
A|B             # A union B
B.union(A, C)   # B union A union C
B|A|C           # B union A union C

# update() – Adds items from another set into the current set.
A.update(B)    
print(A)


# ************ String ************ #
string = 'python is AWesome'    # create a demo string

# capitalize() – Returns a string with the first letter capitalized and all other characters lowercased. It doesn't modify the original string.
print(string.capitalize())

# casefold() – Removes all case distinctions present in a string. It is used for caseless matching, i.e. ignores cases when comparing.
string2 = 'PYTHON IS AWESOME'
print(string2.casefold())  # casefold()
print(string == string2)   # returns False
print(string.casefold() == string2.casefold())    # return True - comparing string and string2 using casefold

# center() – Returns a string which is padded with the specified character.  It doesn't modify the original string.
# The syntax of center() method is: string.center(width[, fillchar])
# when width = length of the string with padded characters and fillchar (optional) = padding character
print(string.center(32))   
print(string.center(32, '*'))      

# count() – Returns the number of occurrences of a substring in the given string.
print(string.count('o'))

# encode() – Returns an encoded version of the given string.  By default, the encode() method does not require any parameters.
# It returns an utf-8 encoded version of the string.  In case of failure, it raises a UnicodeDecodeError exception.
string3 = 'pythön! is awesome!'
print(string3.encode())

# find() – Returns the index of first occurrence of the substring (if found). If not found, it returns -1.
message = 'python is popular programming language'
print(message.find('pop'))
print(message.find('puri'))
print(message.find('pop', 5))
print(message.find('pop', 11, 15))

# partition() – Takes a string parameter separator that separates the string at the first occurrence of it.
print(message.partition('is'))

# replace() – Replaces a specified phrase with another specified phrase. The syntax of the replace() method is:
message2 = "one one was a race horse, two two was one too."
print(message2.replace("one", "three"))

# split() – Breaks up a string at the specified separator and returns a list of strings.
print(message2.split(' '))

# () – Returns a string where the first character in every word is upper case. Like a header, or a title.
print(message2.title())

# zfill() – Adds zeros (0) at the beginning of the string, until it reaches the specified length.
# If the value of the len parameter is less than the length of the string, no filling is done.
txt = "hello"
print(txt.zfill(10))
print(txt.zfill(3))


# ************ from collections import Counter ************ #
# This collections module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.
# Counter is dict subclass for counting hashable objects.  It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values.
# Counts are allowed to be any integer value including zero or negative counts. The Counter class is like bags or multisets in other languages.
from collections import Counter
Counter(['a', 'b', 'c', 'a', 'b', 'b'])
Counter({'a':2, 'b':3, 'c':1})
Counter(a=2, b=3, c=1)


# ************ from itertools import *  ************ #
from itertools import *
# The module that provides various functions that work on iterators to produce complex iterators.
# This module works as a fast, memory-efficient tool that is used either by themselves or in combination to form iterator algebra. 



# ******************************* #
# ----------- Bonus 1 ----------- #
# ******************************* #

flower_orders = ['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
                'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
                'W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
                'R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V',
                'W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y',
                'B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y',
                'W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y',
                'N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y',
                'W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y',
                'W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y',
                'R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O',
                'N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

# 1. Build a counter object and use the counter and confirm they have the same values.
flowerCounter = Counter(flower_orders)
print(flowerCounter)

# 2. Count how many objects have color W in them.
wCount = 0
for flowers in flower_orders:          # loop through flower_orders lists
    myset = set(flowers.split('/'))    # split each flowers_order by '/' to get all letters in the order.  Then make them a set to avoid duplication for each letter.
    for w in myset:
        if w == 'W':
            wCount += 1
print(wCount)

# 3. Make histogram of colors
import matplotlib.pyplot as plt

plt.bar(flowerCounter.keys(), flowerCounter.values())
plt.xticks(rotation='vertical')
plt.show()

# Hint from JohnP - Itertools has a permutation function that might help with these next two.
# Ref: https://docs.python.org/3/library/itertools.html#recipes
# 4. Rank the pairs of colors in each order regardless of how many colors are in an order.
# Instead of using permutations(), I think combinations() suits the purpose of color combination better. 
permList = []
pairList = []

for flowers in flower_orders:           # loop through flower_orders lists
    mylist = flowers.replace('/','')
    permList.append(list(combinations(mylist,2)))
# print(permList)
    
pairList = chain.from_iterable(permList)
pairList = list(pairList)

pairCounter = Counter(pairList)
print(pairCounter)
print(pairCounter.most_common(10))      # Top 10 most common pairs of colors

# 5. Rank the triplets of colors in each order regardless of how many colors are in an order.
trippetsPermList = []
tripetsList = []

for flowers in flower_orders:           # loop through flower_orders lists
    mylist = flowers.replace('/','')
    trippetsPermList.append(list(combinations(mylist,3)))
# print(trippetsPermList)
    
tripetsList = chain.from_iterable(trippetsPermList)
tripetsList = list(tripetsList)

tripetsCounter = Counter(tripetsList)
print(tripetsCounter)
print(tripetsCounter.most_common(10))      # Top 10 most common pairs of colors

# 6. Make dictionary color for keys and values are what other colors it is ordered with.
colorDict = dict(flowerCounter)
print(colorDict)

# 7. Make a graph showing the probability of having an edge between two colors based on how often they co-occur.  (a numpy square matrix)
pairDict = dict(pairCounter)
denominator= sum(pairDict.values())     # find the number of all pairs of colors for denominator
print(denominator)

lists_ofList = []                       
for flowers in flower_orders:           # loop through flower_orders lists to get list of colors
    mylist = list(flowers.split('/')) 
    lists_ofList.append(mylist)
# print(lists_ofList)

unique_color = list({x for l in lists_ofList for x in l})       # find unique colors
unique_color.sort()
print(unique_color)
unique_colorIndex = {k: v for v, k in enumerate(unique_color)}  # assign index numbers to the unique colors, sorted
print(unique_colorIndex)

import numpy as np

matrix = np.zeros(shape=(10,10))        # define empty matrix 10*10 since we have 10 unique colors

for k, v in pairDict.items():           # find the probability of each pair of color based on how often they co-occur
    row = unique_colorIndex.get(k[0])
    col = unique_colorIndex.get(k[1])
    percent = v / denominator * 100
    matrix[row,col] = percent           # assing the probability percent to index that the 2 colors paired
# print(matrix.sum())                   # to check if the total sum of percent is 100%
print(matrix)

# plot the numpy square matrix
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.binary)
plt.colorbar().set_label(label='Probability of Pair Colors (%)',size=10)
plt.xticks(range(len(unique_color)), unique_color)
plt.yticks(range(len(unique_color)), unique_color)
plt.show()

# 8. Make 10 business questions related to the questions we asked above.
## 1) What is the total orders?
## 2) What is the total number of unique color combinations from all orders?
## 3) What are the top seller colors?  What are the least seller colors?
## 4) Are there any colors that have never been sold?
## 5) Are there any two colors that have never sold together?
## 6) Are there any colors that sold as solo color in an order?
## 7) What are the most population combination colors for 2 and 3 colors combination?
## 8) How many colors are in the biggest combination order?
## 9) what are the top 5 color combinations?
## 10) From the questions above, which colors shall I stock more? Which colors shall I lower the stock?



# ******************************* #
# ----------- Bonus 2 ----------- #
# ******************************* #
dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
                        'continent a new nation, conceived in liberty and dedicated to the',
                        'proposition that all men are created equal. Now we are engaged in',
                        'a great civil war, testing whether that nation or any nation so',
                        'conceived and so dedicated can long endure. We are met on a great',
                        'battlefield of that war. We have come to dedicate a portion of',
                        'that field as a final resting-place for those who here gave their',
                        'lives that that nation might live. It is altogether fitting and',
                        'proper that we should do this. But in a larger sense, we cannot',
                        'dedicate, we cannot consecrate, we cannot hallow this ground.',
                        'The brave men, living and dead who struggled here have consecrated',
                        'it far above our poor power to add or detract. The world will',
                        'little note nor long remember what we say here, but it can never',
                        'forget what they did here. It is for us the living rather to be',
                        'dedicated here to the unfinished work which they who fought here',
                        'have thus far so nobly advanced. It is rather for us to be here',
                        'dedicated to the great task remaining before us--that from these',
                        'honored dead we take increased devotion to that cause for which',
                        'they gave the last full measure of devotion--that we here highly',
                        'resolve that these dead shall not have died in vain, that this',
                        'nation under God shall have a new birth of freedom, and that',
                        'government of the people, by the people, for the people shall',
                        'not perish from the earth.']

# 1. Join everything
join_deadMan = ''.join(dead_men_tell_tales)
print(join_deadMan)

# 2. Remove spaces
noSpace_deadMan = join_deadMan.replace(' ', '')
print(noSpace_deadMan)

# 3. Occurrence probabilities for letters
allLetters = ''.join(l for l in noSpace_deadMan if l.isalnum())     # to remove all special characters
allLetters = allLetters.casefold()                                  # regardless capitalization
print(allLetters)
totalLetters = len(allLetters)      # get total number of letters count

import collections

letterCount = dict(Counter(allLetters))          
letterCount = collections.OrderedDict(sorted(letterCount.items()))
print(letterCount)

letterProb = {}
for k, v in letterCount.items():
    percent = v / totalLetters * 100
    prob = {k:percent}
    letterProb.update(prob)

print(letterProb)       # occurrence probabilities for each letter

# 4. Tell me transition probabilities for every letter pairs
pairwiseList = []

def pairwise(iterable):     # pairwise('ABCDEFG') --> AB BC CD DE EF FG, which is perfect to be used for letters transition for me!
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

pairwiseList = list(pairwise(allLetters))
print(pairwiseList)

pairwiseCount = dict(Counter(pairwiseList))
totalPairs = sum(pairwiseCount.values())

letterTransProb = {}
for k, v in pairwiseCount.items():
    percent = v / totalPairs * 100
    prob = {k:percent}
    letterTransProb.update(prob)

print(letterTransProb)      # transition probabilities for each letter pair

# 5. Make a 26x26 graph of 4. in numpy
import string
unique_letter = string.ascii_lowercase
print(unique_letter)
unique_letterIndex = {k: v for v, k in enumerate(unique_letter)}  # assign index numbers to all letters (a-z)
print(unique_letterIndex)

matrixL = np.zeros(shape=(26,26))        # define empty matrix 10*10 since we have 10 unique colors

for k, v in letterTransProb.items():           # find the probability of each pair of color based on how often they co-occur
    row = unique_letterIndex.get(k[0])
    col = unique_letterIndex.get(k[1])
    matrixL[row,col] = v                # assing the probability percent to index that the 2 colors paired
#print(matrixL.sum())                   # to check if the total sum of percent is 100%
print(matrixL)

# 6. plot graph of transition probabilities from letter to letter
# plot the numpy square matrix
plt.imshow(matrixL, interpolation='nearest', cmap=plt.cm.binary)
plt.colorbar().set_label(label='Transition Probabilities (%) \n from Letter on y-axis to Letter on x-axis',size=10)
plt.xticks(range(len(unique_letter)), unique_letter)
plt.yticks(range(len(unique_letter)), unique_letter)
plt.show()

# Unrelated:
# 7. Flatten a nested list
from collections.abc import Iterable

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

L = [[[1, 2, 3], [4, 5]], 6, [7, 8], 9, [10]]

flattenList = list(flatten(L))
print(flattenList)