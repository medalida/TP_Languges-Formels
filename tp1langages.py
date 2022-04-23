#!/usr/bin/env python3
"""
Read a pushdown automaton and a word, returns:
 * ERROR if non deterministic
 * YES if word is recognized
 * NO if word is rejected
"""

from automaton import StackAutomaton, EPSILON, error, warn
import sys
import pdb # for debugging

##################

def is_deterministic(a:'StackAutomaton')->bool:
  trans = a.transitions
  length = len(trans)
  j=0
  for i in range(length):
    for j in range(length):
      if(j<=i): continue
      if(trans[i][1]!=trans[j][1]): continue #check the source;
      if(trans[i][2]==trans[j][2] and trans[i][3]==trans[j][3]): #check 1st rule;
        return False 
      if((trans[i][2]=='%' or trans[j][2]=='%') and trans[i][3]==trans[j][3]): #check 2nd and 3rd rule;
        return False
  return True

  
##################
  
def recognizes(a:'StackAutomaton', word:str)->bool:

  stack = []
  word_stack = list(word[len(word)::-1]) #reverse the word then transforme it to a stack;
  stack.append(a.initial_stack)
  state = a.initial.name

  while(len(word_stack)!=0):
    ALPHA = stack.pop()
    alpha = word_stack.pop()
    find = False

    for (source, letter, head, push, dest) in a.transitions:
      if(source != state): continue #check the source;
      if(head != ALPHA): continue #check the stack letter;
      if(letter != alpha and letter != '%'): continue #check the letter;
      if(letter == '%'): word_stack.append(alpha) #check if it is a transition of epsilon; -> excute_epsilon();
      stack = stack + push[::-1]
      state = dest
      find = True
      break
    
    if(not find): return False

  if(state in a.acceptstates): return True
  return False

##################

if __name__ == "__main__" :
  if len(sys.argv) != 3:
    usagestring = "Usage: {} <automaton-file.ap> <word-to-recognize>"
    error(usagestring.format(sys.argv[0]))

  automatonfile = sys.argv[1]  
  word = sys.argv[2]

  a = StackAutomaton("dummy")
  a.from_txtfile(automatonfile)
  print(a)
  if not is_deterministic(a) :
    print("ERROR")
  elif recognizes(a, word):
    print("YES")
  else:
    print("NO")

