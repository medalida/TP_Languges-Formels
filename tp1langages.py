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
  #TODO implement!
  return False
  
##################
  
def recognizes(a:'StackAutomaton', word:str)->bool:
  #TODO implement!
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

