from pprint import pprint

# The productions rules have to be binarized.

# grammar_text = """
# S -> NP VP
# NP -> Det Noun
# VP -> Verb NP
# PP -> Prep NP
# NP -> NP PP
# VP -> VP PP
# """

# lexicon = {
#     'Noun': set(['cat', 'dog', 'table', 'food']),
#     'Verb': set(['attacked', 'saw', 'loved', 'hated']),
#     'Prep': set(['in', 'of', 'on', 'with']),
#     'Det': set(['the', 'a']),
# }
grammar_text = """
S -> NPZ VP
S -> NP VBZ
NP -> Det Noun
NPZ -> Det Nouns
VP -> Verb NP
VBZ -> Verbs NP
PP -> Prep NP
NP -> NP PP
VP -> VP PP
"""

lexicon = {
    'Nouns': set(['cats', 'dogs']),
    'Verbs': set(['attacks', 'attacked']),
    'Noun': set(['cat', 'dog', 'table', 'food']),
    'Verb': set(['saw', 'loved', 'hated', 'attack']),
    'Prep': set(['in', 'of', 'on', 'with']),
    'Det': set(['the', 'a']),
}

# Process the grammar rules.  You should not have to change this.
grammar_rules = []
for line in grammar_text.strip().split("\n"):
    if not line.strip(): continue

    left, right = line.split("->")
    left = left.strip()
    children = right.split()
    rule = (left, tuple(children))
    # print "The Rule", rule , type(rule)
    grammar_rules.append(rule)
possible_parents_for_children = {}
for parent, (leftchild, rightchild) in grammar_rules:
    if (leftchild, rightchild) not in possible_parents_for_children:
        possible_parents_for_children[leftchild, rightchild] = []
    possible_parents_for_children[leftchild, rightchild].append(parent)
# Error checking
all_parents = set(x[0] for x in grammar_rules) | set(lexicon.keys())
for par, (leftchild, rightchild) in grammar_rules:
    if leftchild not in all_parents:
        assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % leftchild
    if rightchild not in all_parents:
        assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % rightchild

# print "Grammar rules in tuple form:"
# pprint(grammar_rules)
# print "Rule parents indexed by children:"
# pprint(possible_parents_for_children)


def cky_acceptance(sentence):
    # return True or False depending whether the sentence is parseable by the grammar.
	global grammar_rules, lexicon

	# Set up the cells data structure.
	# It is intended that the cell indexed by (i,j)
	# refers to the span, in python notation, sentence[i:j],
	# which is start-inclusive, end-exclusive, which means it includes tokens
	# at indexes i, i+1, ... j-1.
	# So sentence[3:4] is the 3rd word, and sentence[3:6] is a 3-length phrase,
	# at indexes 3, 4, and 5.
	# Each cell would then contain a list of possible nonterminal symbols for that span.
	# If you want, feel free to use a totally different data structure.
	# print type(possible_parents_for_children)
	N = len(sentence)
	cells = {}
	# print "sentence", sentence, type(sentence)
	
	for i in range(N):
	    for j in range(i + 1, N + 1):
	        cells[(i, j)] = []
	

	count =0
	for num in range(0,N):
		for i in range(0,N-count):
			if(count==0):
				for lex in lexicon:
					if sentence[i] in lexicon[lex]:
						cells[(i,i+1)].append(lex)
			else:
				# print "Entering curr"
				for curr in range (i+1,i+count+1):
					# these return lists ok so now we have to use matching
					leftcell= cells[(i,curr)]
					rightcell=cells[(curr, i+count+1)]
					# Assuming that the lists are not empty
					for l in leftcell:
						for r in rightcell:
							# if the key exists
							# print "l",l,"r",r
							# print " "
							if (l,r) in possible_parents_for_children:
								# find the value and append it
								j=i+count+1
								# print type(cells[(i,j)]), cells[(i,j)], "i", i, "j", j
								lrparent=possible_parents_for_children[(l,r)]
								# print lrparent,"lrparent" #this is returning a list
								cells[(i,j)].extend(lrparent)
			if(i==(N-count-1)):
				count=count+1

	
	pprint(cells)

	if cells[(0,N)] == ['S']:
		return True 
	else:
		return False



def cky_parse(sentence):
	# Return one of the legal parses for the sentence.
	# If nothing is legal, return None.
	# This will be similar to cky_acceptance(), except with backpointers.

	global grammar_rules, lexicon

	# print "Grammar rules\n", grammar_rules, "\n Lexicon\n", lexicon, "\n\n"
	N = len(sentence)
	cells = {}
	for i in range(N):
	    for j in range(i + 1, N + 1):
	        cells[(i, j)] = []



	# TODO replace the below with an implementation


	# Try 2

	count =0
	a = range(5, 500)
	for num in xrange(100):
		for a in range(N):
			for i in range(0,N-count):
				#print "Inside the i Loop",i, N-count
				# pprint (cells)
				if(count==0):
					for lex in lexicon:
						if sentence[i] in lexicon[lex]:
							# not solid about this
							# will have to fix this append too maybe have a special case to recognize that we are ay yeh root
							cells[(i,i+1)].append([lex,-1,sentence[i],sentence[i]])
					# print N-count-1, i
					if (i== N-count-1):
						#print "REached here", N-count-1, i
						count=count+1
						#print "count is ", count
				else:
					# when count is not 0 you are now building the tree
					# print "We enter the loop"
					for curr in range(i+1,i+count+1):
						# List of tuples, can also be empty, check how appending two empty lists works
						leftcell=cells[(i,curr)]
						#print "i",i, "j",(i+count+1),"curr",curr
						rightcell=cells[(curr,i+count+1)]

						# print " "
						# print "Left Cell",type(leftcell),leftcell
						# print " "
						# print "Right Cell",type(rightcell),rightcell
						# will this break if the lists are not empty
						for l in leftcell:
							for r in rightcell:
								# if the key exists 
								# this will break you have to refine it
								# ln, rn storing the nominals 
								ln=l[0]
								rn=r[0]
								# print "Before check", ln, rn, "\n\n"	
								import types
								from six import string_types
								try:
									if not isinstance(ln, string_types):
										while isinstance(ln, (list,)):
											ln = ln[0]
								except:
									import sys 
									print "left->", sys.exc_info(), ln
								try:
									if not isinstance(rn, string_types):
										while isinstance(rn, (list,)):
											rn = rn[0]
								except:
									print "right->", sys.exc_info(), rn
								# print "Final check", ln, rn
								# print possible_parents_for_children
								if (ln, rn) in possible_parents_for_children:
									j=i+count+1
									# print type(cells[(i,j)]), cells[(i,j)], "i", i, "j", j
									lrparent=possible_parents_for_children[(ln,rn)][0]
									cells[(i,j)].append([lrparent,curr,ln,rn])
				if(i==(N-count-1)):
					count=count+1
					# print "Count Incremented"
					#'''
		#print "End", num, "Num", N, count

	pprint (cells)
	# returning null
	# return cells[(0,N)]
	if not cells[(0,N)]:
		return None
	else:
		# parse tree 
		# initiate a list of lists call it TREE
		tree=[]
		print "Printing parse tree"
		pprint(cells)
		parse_tree(0,N,tree,cells)
		#print tree
		return tree
		
    
def parse_tree(start,end,tree,cells):
	# print cells[(start,end)][0] ,"Cell Value"
	#print "I am inside parseTree "
	# print "start",start,"end",end
	Nonterminal, curr, x,y=cells[(start,end)][0]
	# print "Checking data", Nonterminal, curr, x, y
	# base case:
	tree.append(Nonterminal)
	# print "Tree -> ", tree
	if curr != -1:
		tree.append([])
		tree.append([])
		left_rec= parse_tree(start,curr,tree[1],cells)
		right_rec= parse_tree(curr,end,tree[2],cells)
	# cannot append lists this way
	#return tree.append([Nonterminal,[left_rec,right_rec]])

    
# print cky_acceptance(['the','cat','attacked','the','food'])
# pprint( cky_parse(['the','cat','attacked','the','food']))
# pprint( cky_acceptance(['the','the']))
# pprint( cky_parse(['the','the']))
# print cky_acceptance(['the','cat','attacked','the','food','with','a','dog'])
# pprint( cky_parse(['the','cat','attacked','the','food','with','a','dog']) )
# pprint( cky_parse(['the','cat','with','a','table','attacked','the','food']) )