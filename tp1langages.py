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
      if(trans[i][0]!=trans[j][0]): continue #check the source;
      if(trans[i][1]==trans[j][1] and trans[i][2]==trans[j][2]): #check 1st rule;
        return False 
      if((trans[i][1]=='%' or trans[j][1]=='%') and trans[i][2]==trans[j][2]): #check 2nd and 3rd rule;
        return False
  return True

def excute_epsilon(a:'StackAutomaton', p:str, w:list):
  ALPHA = ''
  while(len(w)>0): #check if it is a transition of epsilon;
    
    ALPHA = w.pop()
    find = False
    for (source, letter, head, push, dest) in a.transitions:
      if(source != p): continue #check the source;
      if(head != ALPHA): continue #check the stack letter;
      if(letter != '%'): continue #check the letter;
      w = w + push[::-1]
      p = dest
      find = True
    if(not find):
      w.append(ALPHA)
      return p
  return p

  
##################
  
def recognizes(a:'StackAutomaton', word:str)->bool:
  word=word.replace("%","")
  stack = []
  word_stack = list(word[len(word)::-1]) #reverse the word then transforme it to a stack;
  stack.append(a.initial_stack)
  state = a.initial.name
  while(len(word_stack)!=0 or len(stack)!=0):
    state = excute_epsilon(a, state, stack)
    if(len(word_stack)==0): 
      if(state in a.acceptstates): return True
      else: return False
    ALPHA = stack.pop()
    alpha = word_stack.pop()
    find = False
    
    for (source, letter, head, push, dest) in a.transitions:
      if(source != state): continue #check the source;
      if(head != ALPHA): continue #check the stack letter;
      if(letter != alpha): continue #check the letter;
      stack = stack + push[::-1]
      state = dest
      find = True
      if(len(word_stack)==0 and state in a.acceptstates): return True
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

