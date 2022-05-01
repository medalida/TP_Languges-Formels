#!/usr/bin/env python3
"""
Read a grammar and a word, returns:
 * YES if word is generated
 * NO otherwise
"""

from automaton import Grammar, EPSILON, error, warn
import sys
import pdb # for debugging

  
##################
  
def generates(a:'Grammar', word:str)->bool:
  word=word.replace("%","")
  if(word==''): #check epsilon;
    return((''+a.initial.name,[]) in a.rules)
  length = len(word)
  matrix = [ [ [] for i in range(length+1) ] for j in range(length+1)] #create empty matrix;

  for i in range(length): #fill the bottom of the matrix;
    for (left, right) in a.rules:
      if(word[i] in right):
        matrix[i][i+1].append(left)
  
  for i in range(2,length+1): #iterate lines;
    for j in range(length-i+1): #iterate cells;
      k=j+i

      for l in range(j+1,k): #iterate the cell to find the factors;
        for let1 in matrix[j][l]: #iterate the cell of the first factor;
          for let2 in matrix[l][k]: #iterate the cell of the second factor;
            for (left, right) in a.rules: #iterate rules;

              if(let1+let2 == ''.join(right)): #compare the tow factors to the rule;
                matrix[j][k].append(left) #if true, add the left side to the cell;


  return (a.initial.name in matrix[0][length]) #check if S in the last cell;

##################

if __name__ == "__main__" :
  if len(sys.argv) != 3:
    usagestring = "Usage: {} <grammar-file.gr> <word-to-recognize>"
    error(usagestring.format(sys.argv[0]))

  grammarfile = sys.argv[1]  
  word = sys.argv[2]

  a = Grammar("dummy")
  a.from_txtfile(grammarfile)
  print(a)

  if generates(a, word):
    print("YES")
  else:
    print("NO")

