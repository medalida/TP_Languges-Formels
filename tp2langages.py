#!/usr/bin/env python3
"""
Read a grammar, checks if it is in Chomsky normal form and puts it in Chomsky normal form if not.
"""

from automaton import Grammar, EPSILON, error, warn
import sys
import pdb # for debugging

##################

def is_CNF(a:'Grammar')->bool:
  #TODO implement!
  return False
  
##################
  
def to_CNF(a:'Grammar'):
  step1_CNF(G)
  step2_CNF(G)
  step3_CNF(G)
  step4_CNF(G)
  step5_CNF(G)

##################
  
def step1_CNF(a:'Grammar'):
  #TODO implement!
  print("step 1 done")


##################

def step2_CNF(a:'Grammar'):
  #TODO implement!
  print("step 2 done")


##################

def step3_CNF(a:'Grammar'):
  #TODO implement!
  print("step 3 done")


##################

def step4_CNF(a:'Grammar'):
  #TODO implement!
  print("step 4 done")


##################

def step5_CNF(a:'Grammar'):
  #TODO implement!
  print("step 5 done")

##################

if __name__ == "__main__" :
  if len(sys.argv) != 2:
    usagestring = "Usage: {} <grammar-file.gr>"
    error(usagestring.format(sys.argv[0]))

  grammarfile = sys.argv[1]  

  G = Grammar("dummy")
  G.from_txtfile(grammarfile)
  print(G)

  if is_CNF(G):
    print("grammar already in CNF")
  else:
    to_CNF(G)

