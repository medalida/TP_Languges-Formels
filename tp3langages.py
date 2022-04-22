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
  #TODO implement!
  return False

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

