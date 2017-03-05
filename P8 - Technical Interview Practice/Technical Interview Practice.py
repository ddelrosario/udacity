
# coding: utf-8

# ## Technical Interview Practice - Dominic Del Rosario
# 
# 
# 
# #### For question 1, the steps I took are:
# 
# 1  First, I took the string t.  
# 
# 2  Then I checked each character of t to see if it's in s.  If so, count it.
# 
# 3  If the number of characters in t that are in s equal the length of t, then it's an anagram.
# 
# Efficiency:
# Time complexity will be O(n) because it run according to the length of the strings s & t.  It is a linear relationship.
# Space complexity will be O(1) because the number of anagrams are finite regardless of the size of the input.

# In[358]:

'''
Question 1
Given two strings s and t, determine whether some anagram of t is a substring of s. 
For example: if s = "udacity" and t = "ad", then the function returns True. Your function 
definition should look like: question1(s, t) and return a boolean True or False.

'''

def question1(s, t):
    
    word_count = 0 
    for letters in t:
        if (letters in s): 
            word_count += 1
    if (s == '' or t == ''):
        word_count = -1

    if word_count == len(t):
        return True
    else:
        return False

s0 = ''
t0 = ''

s1 = 'blah'
t1 = 'lb'

s2 = 'udacity'
t2 = 'ad'

s3 = 'udacity'
t3 = 'ytci'

s4 = 'udacity'
t4 = 'bob'

print 'Test Cases for Question 1:'

print question1(s0, t0)
# Should print False

print question1(s1, t1)
# Should print True

print question1(s2, t2)
# Should print True

print question1(s3, t3)
# Should print True

print question1(s4, t4)
# Should print False


# In[ ]:




# #### For question 2, the steps I took are:
# 
# 1  First I had to think what the definition of a palindrome is.
# 
# 2  Then, I created a function to check if a word is a palindrome.
# 
# 3  Then, I iterated through every combination of "words" in the string and 
#    if it is a palindrome, it will return it.
# 
# Efficiency:
# Time efficiency would be O(n^2) because I am using a nested operation.   I am iterating to find every combination of words.
# Space efficiency would be O(1) since I am looking for one palindrome.

# In[359]:

'''
Question 2
Given a string a, find the longest palindromic substring contained in a. Your function definition 
should look like question2(a), and return a string.

'''
def isPalindrome(word):
    if word == word[::-1]:
        return True
    else:
        return False
    
def question2(a):
    
    if a == '':
        return None
    else:
        for i in range(len(a), 1, -1):
            for j in range(len(a), i-1, -1):
                if isPalindrome(a[(j-i):j]):
                    return a[(j-i):j]
    

print 'Test Cases for Question 2:'

word0 = ''
print question2(word0)
# Should print None

word1 = 'registerr'
print question2(word1)
# Should print "rr"

word2 = 'bob'
print question2(word2)
# Should print "bob"

word3 = 'tattarrattat'
print question2(word3)
# Should print "tattarrattat"

word4 = 'registers'
print question2(word4)
# Should print None


# In[ ]:




# #### For question 3, the steps I took are:
# 
# 1  I will start off by saying I know this is the not the most efficient or elegant way to solve this but I believe I have a solution.
# 
# 2  I am using Kruskal's algorithm.   So I start by sorting all the edges by weight.
# 
# 3  Then I only keep unique edges.
# 
# 4a  Going through the algorithm, I add each edge until done.  Done means until the number of vertices - 1.
# 
# 4b  I created several if statements to check if adding an edge will result in a cycle.  If not, it adds the edge.
# 
# 5  Then I print the adjacency list.
# 
# Efficiency:
# 
# Time efficiency should still be O(n^2) because I am using nested conditions.   I am iterating through the edges.
# Although, I am using multiple for loops, it should be at worst O(n^2).
# Space efficiency would be O(n) since I am creating arrays and sets to store the edges.  As the number of nodes increases, the number of edges increases and therefore the arrays and sets all increase.

# In[360]:

'''
Question 3
Given an undirected graph G, find the minimum spanning tree within G. A minimum spanning tree connects all vertices 
in a graph with the smallest possible total weight of edges. Your function should take in and return an adjacency list 
structured like this:

{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5)]}
 
Vertices are represented as unique strings. The function definition should be question3(G)
'''
def question3(G):
    vertices = sorted(G.keys())

    #kriskal's algorithm
    make_edge_list = []
    make_set = set()
    min_span_list = []
    adjacency_list = []

    #create a list of the edges and weights
    for i in G:
        for j in G[i]:
            make_edge_list.append((i,j))
    #print make_edge_list        

    #create the set of unique edges
    for edge in make_edge_list:
        if ((edge[1][0], (edge[0], edge[1][1])) or
            (edge[0], (edge[1][0], edge[1][1]))) not in make_set:
            make_set.add(edge)

    #sort the unique edges    
    sorted_make_set = sorted(make_set, key=lambda edge: edge[1][1])    
    #print sorted_make_set, len(sorted_make_set)

    #check for cycles & stop until we reach vertices - 1
    stop = len(vertices) - 1
    span_count = 0

    temp_node_list1 = set()
    temp_node_list2 = set()

    #add edge 1
    temp_node_list1.add(sorted_make_set[0][0])
    temp_node_list1.add(sorted_make_set[0][1][0])        
    min_span_list.append(sorted_make_set[0])

    for edges in sorted_make_set:
        
        isCycle = True
        
        #if vertices already in, skip
        if ((edges[0] in temp_node_list1 and edges[1][0] in temp_node_list1)):
            isCycle = True
        
        #if vertices in both sets, union
        ###SOMETHING WRONG HERE
        elif ((edges[0] in temp_node_list1 and edges[1][0] in temp_node_list2) or
           (edges[1][0] in temp_node_list1 and edges[0] in temp_node_list2)):
            temp_node_list1 = temp_node_list1.union(temp_node_list2)
            isCycle = False
        
        #if only one vertex in set, add
        elif ((edges[0] not in temp_node_list1 and edges[1][0] in temp_node_list1) or
        (edges[0] in temp_node_list1 and edges[1][0] not in temp_node_list1)):
            temp_node_list1.add(edges[0])
            temp_node_list1.add(edges[1][0])    
            isCycle = False
        
        #if no vertices, add to temp set
        elif edges[0] not in temp_node_list1 and edges[1][0] not in temp_node_list1:
            temp_node_list2.add(edges[0])
            temp_node_list2.add(edges[1][0])    
            isCycle = False

        #add edge is not a cycle and stop at vertices - 1
        if span_count < stop and isCycle is False:
            min_span_list.append(edges)
            span_count += 1
       
    #print the adjacency list
    print ""
    for span in min_span_list:
        adjacency_list.append((span[0], span[1]))
        adjacency_list.append((span[1][0], (span[0], span[1][1])))
    return sorted(adjacency_list)

print 'Test Cases for Question 3:'

print question3({'A': [('B', 2)],
     'B': [('A', 2), ('C', 5)], 
     'C': [('B', 5)]})
#should print
#[('A', ('B', 2)), ('B', ('A', 2)), ('B', ('C', 5)), ('C', ('B', 5))]

print question3({'A': [('B', 8), ('D', 5), ('C', 6)],
     'B': [('A', 8), ('D', 4)], 
     'C': [('A', 6), ('D', 3)], 
     'D': [('C', 3), ('A', 5), ('B', 4)]})
#should print
#[('A', ('D', 5)), ('B', ('D', 4)), ('C', ('D', 3)), ('D', ('A', 5)), ('D', ('B', 4)), ('D', ('C', 3))]

print question3({'A': [('E', 5), ('H', 6), ('B', 8), ('F', 1)],
     'B': [('A', 8), ('C', 4), ('F', 6)],
     'C': [('B', 4), ('F', 2), ('G', 7)],
     'E': [('A', 5), ('H', 3)],
     'F': [('A', 1), ('B', 6), ('C', 2), ('G', 9), ('H', 5)],     
     'G': [('F', 9), ('C', 7)],
     'H': [('E', 3), ('A', 6), ('F', 5)]})
#should print
#[('A', ('F', 1)), ('B', ('C', 4)), ('C', ('B', 4)), ('C', ('F', 2)), ('C', ('G', 7)), ('E', ('H', 3)), 
#('F', ('A', 1)), ('F', ('C', 2)), ('F', ('H', 5)), ('G', ('C', 7)), ('H', ('E', 3)), ('H', ('F', 5))]


# In[ ]:




# #### For question 4, the steps I took are:
# 
# 1  Start from root
# 
# 2  If the node is between n1 & n2 in different subtrees, we have the least common ancestor (LCA).
# 
# 3  If the n1 & n2 are in the right subtree and greater than the node, we need to move down the tree until we have the LCA.
#    According to wikipedia, a node can be descendent of itself.
#    https://en.wikipedia.org/wiki/Lowest_common_ancestor
# 
# 4  Otherwise, we need to move down the left side of the tree until we have the LCA.
# 
# Efficiency:
# 
# Time complexity should be O(height*n) since we are just working down the tree.   As n grows, it does linearly.
# Space complexity would be O(1) since I am only using the input linked list.

# In[89]:

'''
Question 4
Find the least common ancestor between two nodes on a binary search tree. The least common ancestor is the farthest 
node from the root that is an ancestor of both nodes. For example, the root is a common ancestor of all nodes on the 
tree, but if both nodes are descendents of the root's left child, then that left child might be the lowest common 
ancestor. You can assume that both nodes are in the tree, and the tree itself adheres to all BST properties. 
The function definition should look like question4(T, r, n1, n2), where T is the tree represented as a matrix, 
where the index of the list is equal to the integer stored in that node and a 1 represents a child node, r is a 
non-negative integer representing the root, and n1 and n2 are non-negative integers representing the two nodes in 
no particular order. For example, one test case might be

question4([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]],
          3,
          1,
          4)
and the answer would be 3.

'''

def question4(T, r, n1, n2):
    
    #least common ancestor
    LCA = r
    left = 0
    right = 0
    
    if len(T) == 0:
        return None
    elif len(T) == 1:
        return r
    else:
        while not (n1 <= LCA and LCA <= n2):
            if n1 <= LCA:
                LCA = T[LCA].index(1,0)
            else:
                LCA = T[LCA].index(1,LCA)
        return LCA
    

print 'Test Cases for Question 4:'

print question4([],
                None,
                None,
                None)
# Should print None

print question4([[0]],
                0,
                0,
                0)
# Should print 0

print question4([[0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]],
                3,
                1,
                4)
# Should print 3
                   
print question4([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                3,
                4,
                6)
# Should print 5

print question4([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0]],
                1,
                4,
                6)
# Should print 4

print question4([[0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,1,0,0,0],
                 [0,0,1,0,0],
                 [0,0,0,1,0]],
                4,
                1,
                3)
# Should print 3


# In[ ]:




# #### For question 5, the steps I took are:
# 
# 1  I start by counting until node.next is NULL
# 
# 2  Then I get the length of the linked list
# 
# 3  Finally, I iterate until I get to (length - m)
# 
# Efficiency:
# 
# Time complexity should be O(n).  I am doing computations on the list and iterating through the list.   If the list was larger, it should be a linear relationship.
# Space complexity would be O(n), since I am only assigning variables.   As the linked list grows, the amount of memory needed should only grow linearly.

# In[362]:

'''
Question 5
Find the element in a singly linked list that's m elements from the end. For example, if a linked list has 5 elements, 
the 3rd element from the end is the 3rd element. The function definition should look like question5(ll, m), 
where ll is the first node of a linked list and m is the "mth number from the end". You should copy/paste the 
Node class below to use as a representation of a node in the linked list. Return the value of the node at that position.

class Node(object):
  def __init__(self, data):
    self.data = data
    self.next = None
    
'''

class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None

def question5(ll, m):
    if ll:
        length = 1
        node = ll
        # compute the length of the linked list
        while node.next:
            node = node.next
            length += 1

        # iterate until you get to (length - m)
        if m < length:
            i = 0
            node2 = ll
            while i < (length - m):
                node2 = node2.next
                i += 1
        else:
            return None
    else:
        node = ll
    return node2.data

print 'Test Cases for Question 5:'

d0 = Node(0)
print question5(d0,8)
#should print None

e0 = Node(0)
e1 = Node(1)
e2 = Node(2)
e3 = Node(3)
e4 = Node(4)
e5 = Node(5)
e6 = Node(6)
e7 = Node(7)
e8 = Node(8)
e9 = Node(9)
e10 = Node(10)
e0.next = e1
e1.next = e2
e2.next = e3
e3.next = e4
e4.next = e5
e5.next = e6
e6.next = e7
e7.next = e8
e8.next = e9
e9.next = e10
print question5(e0, 3)
# Should print 8

f0 = Node(0)
f1 = Node(1)
f2 = Node(2)
f3 = Node(3)
f4 = Node(4)
f5 = Node(5)
f0.next = f1
f1.next = f2
f2.next = f3
f3.next = f4
f4.next = f5
print question5(f0, 2)
# Should print 4


# In[ ]:



