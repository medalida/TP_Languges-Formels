#!/usr/bin/env python3
"""
Module to represent, build and manipulate finite state automata
"""

from typing import Dict, List, Union, Tuple, Optional
from collections import OrderedDict, Counter # remember order of insertion
import sys
import os.path
import pdb
import re

  

########################################################################
########################################################################

def warn(message, *, warntype="WARNING", pos="", **format_args):
  """Print warning message."""
  msg_list = message.format(**format_args).split("\n")
  beg, end = ('\x1b[31m', '\x1b[m') if sys.stderr.isatty() else ('', '')
  if pos: pos += ": "
  for i, msg in enumerate(msg_list):
    warn = warntype if i==0 else "."*len(warntype)
    print(beg, pos, warn, ": ", msg, end, sep="", file=sys.stderr)

##################

def error(message, **kwargs):
    """Print error message and quit."""
    warn(message, warntype="ERROR", **kwargs)
    sys.exit(1)
    
########################################################################
########################################################################

try: # Make the library robust
  from graphviz import Source
except ModuleNotFoundError:
  warn("Warning: graphviz not installed, will not draw automaton graphically")
  class Source: # Dummy class for typing only
    def __init__(self,res):
      pass
    def render(self,outfilename):
      warn("Graphviz not installed, cannot draw automaton")

EPSILON = "%" # Constant to represent empty string

########################################################################

class State(object):
  """
  Represents a state in the automaton, with its list of transitions
  """
  transitions: Dict[str,Dict['State',None]]
  name: str
  is_accept: bool
  
##################

  def __init__(self, name:str)->None:
    """
    Create a new state with a given name
    """
    self.name = name
    self.transitions = OrderedDict()  # by default, empty dict
    self.is_accept = False # by default, create non-accept state

##################
    
  def make_accept(self, accepts:bool=True):
    """
    Transform state into accept state, or the opposite, if accepts=False
    """
    self.is_accept = accepts

##################
   
  def add_transition(self, symbol:str, dest:'State'):
    """
    Add a transition on string `symbol` to State `dest`
    """
    destset = self.transitions.get(symbol,OrderedDict())
    if dest in destset:
      warn("Redundant transition: {s} -{a}-> {d}",s=self.name,
                                                  a=symbol,d=dest.name)
    destset[dest]=None
    self.transitions[symbol] = destset

##################

  def __str__(self)->str:
    """
    Standard function to obtain a string representation of a state
    """
    return self.name.replace('"',"&quot;")

########################################################################
########################################################################

class Automaton(object):
  """
  An automaton is a list of states and a pointer to the initial state
  Transitions and acceptance are represented inside states (see above)
  """  
  name:str
  statesdict:Dict[str,State]
  initial:State
  
##################  
  
  def __init__(self,name:str)->None:
    self.reset(name)     

##################

  def reset(self,name:str=None):
    """
    Reinitialize the automaton with empty content
    """
    if name:
      self.name = name
    elif not self.name:
      self.name = "NewAutomaton"
      warn("New automaton has no name")
    self.statesdict = OrderedDict()   
    self.initial = None 

##################

  def deepcopy(self)->'Automaton':
    """
    Make a deep copy of itself, that is a copy of all elements
    """
    a = Automaton(self.name + "_copy")
    a.from_txt(self.to_txtfile())
    return a

##################
      
  def add_transition(self, src:str, symbol:str, dst:str):
    """
    Add a transition from `src` to `dst` on `symbol`
    """    
    src_state = self.statesdict.get(src, State(src)) # create if absent    
    if not self.statesdict and not self.initial: 
      self.initial = src_state # by default first state added is initial
    self.statesdict[src] = src_state # if new, add to dict
    dst_state = self.statesdict.get(dst, State(dst)) # create if absent
    self.statesdict[dst] = dst_state # if new, add to dict
    src_state.add_transition(symbol, dst_state) # add the transition    

##################

  def remove_state(self, delstate:str):
    """
    Remove `delstate` from the automaton as well as all its transitions.
    """
    if delstate not in self.statesdict :     
      warn("State {} does not exist, will not remove".format(delstate))
      return    
    delstateobj = self.statesdict[delstate]
    del(self.statesdict[delstate]) # First, remove from main states list
    for state in self.statesdict.values(): # For all remaining states
      for (symbol,dests) in state.transitions.items() :
        if delstateobj in dests :
          del(dests[delstateobj])
      if state.transitions and not state.transitions[symbol] : 
        del(state.transitions[symbol]) # no transition left on symbol
          
##################

  def remove_transition(self, src:str, symbol:str, dst:str):
    """
    Remove a transition from `src` to `dst` on `symbol`
    """ 
    try:
      del(self.statesdict[src].transitions[symbol][self.statesdict[dst]])
    except KeyError:
      warn("Transition {} -{}-> {} not found".format(src,symbol,dst))

##################

  @property
  def states(self) -> List[str]:
    return list(self.statesdict.keys())

##################

  @property
  def alphabet(self)->List[str]:
    """
    Get the set of symbols used in the current transitions (+ epsilon)
    """
    alphabet:Dict[str,None] = OrderedDict()
    for state in self.statesdict.values():
      for s in state.transitions.keys():       
        alphabet[s] = None
    return list(alphabet.keys())

##################

  @property
  def reachable_states(self) -> List[str]:
    """
    Returns a list of reachable states in the automaton
    """
    result = set([self.initial])
    nbelems = 0
    while nbelems != len(result) :
      nbelems = len(result)
      addtoresult = []
      for state in result :        
        for destlist in state.transitions.values() :
          for dest in destlist :
            addtoresult.append(dest)
      result = result.union(addtoresult)
    return list(map(lambda x:x.name,result))

##################

  def remove_unreachable(self):
    """
    Remove unreachable states from the automaton
    """
    removed = []
    for state in self.states :
      if not state in self.reachable_states :
        removed.append(state)
    for r in removed :
      self.remove_state(r)

##################

  @property
  def acceptstates(self)->List[str]:
    """
    Return a set of accept states in the automaton.
    If the initial state is accepting, then it is the first in the list
    """
    accept = OrderedDict({k:None for (k,v) in self.statesdict.items() \
                                 if v.is_accept})
    if self.initial and self.initial.name in accept :
      result = [self.initial.name]
      del(accept[self.initial.name])
    else :
      result = []
    return result + list(accept.keys())
    
##################

  @property
  def transitions(self)->List[Tuple[str,str,str]]:
    """
    Returns a list of transitions, each represented as a tuple.
    The tuple contains three strings: (source, symbol, destination)
    The first transitions are always from the initial state
    """
    result = []
    # Transitions from initial state    
    for (symbol,dests) in self.initial.transitions.items():
      for destination in dests :
        result.append((self.initial.name,symbol,destination.name))       
    # Transitions from other states      
    for source in self.statesdict.values():
      if source != self.initial :
        for (symbol,dests) in source.transitions.items():
          for destination in dests :
            result.append((source.name,symbol,destination.name))
    return result
          

##################

  def rename_state(self, oldname: str, newname: str):
    """
    Renames a state in the automaton from `oldname` to `newname`
    """
    if newname in self.states :
      warn("New name \"{}\" already exists, try a new one".format(newname))
      return
    try :
      self.statesdict[newname] = self.statesdict[oldname]
      self.statesdict[newname].name = newname
      del self.statesdict[oldname]
    except KeyError:
      warn("Tried to rename not-existent state \"{}\"".format(oldname))

##################

  @property
  def transition_table(self)->str:
    """
    Return a string representing the transition table of the automaton
    """
    res = ""
    rows = [[""]+self.alphabet]
    maxlen = 1
    for s in self.statesdict.values():
      row = [s.name]
      for a in rows[0][1:] : # for every symbol of the alphabet
        dest = s.transitions.get(a, None)
        if dest and len(dest) == 1:
          row.append(list(dest)[0].name)          
        elif dest: # non-deterministic
          row.append("{"+",".join([x.name for x in dest])+"}")
        else:
          row.append(" ")
        maxlen = max(maxlen,len(row[-1])) # maximum length of a cell
      rows.append(row)    
    for row in rows:      
      res += "|"+"|".join([("{:"+str(maxlen)+"}").format(c) for c in row])+"|\n"
      res += "-"*((maxlen+1)*len(row)+1) + "\n"
    return res[:-1]
	  
##################
    
  def make_accept(self, src:Union[str,List[str]], accepts:bool=True):
    """
    Transform the a state(s) of the automaton into accept state(s)
    """
    if isinstance(src,str):
      src = [src] # transform in list if necessary
    for srci in src:
      if srci not in self.statesdict:
        error("Accept state {a} inexistent!",a=srci)
      self.statesdict[srci].make_accept(accepts)
    
##################

  def __str__(self)->str:
    """
    Standard function to obtain a string representation of an automaton
    """
    tpl = "{A} = <Q={{{Q}}}, S={{{S}}}, D, q0={q0}, F={{{F}}}>\nD =\n{D}"    
    return tpl.format(A=self.name, Q=str(",".join(self.states)),
                      S=",".join(self.alphabet), q0=self.initial, 
                      F=",".join(self.acceptstates),
                      D=self.transition_table)

##################
    
  def to_graphviz(self, outfilename:str=None) -> Source:
    if not self.states:
      tpl = "digraph L{{label=\"{name}\"; node [shape=record]; a [label=\"empty\"]}}"
      res = tpl.format(name=self.name)
    else:
      res = """digraph finite_state_machine {
  rankdir=LR;  
  size=\"8,5\""""
      res += "  label=\"{}\"".format(self.name)
      if self.acceptstates:
        accept = " ".join(map(lambda x : '"'+x+'"', self.acceptstates))
        res += "  node [shape = doublecircle]; {};\n".format(accept)   
      res += "  node [shape = circle];\n"
      res += "  __I__ [label=\"\", style=invis, width=0]\n"
      res += "  __I__ -> \"{}\"\n".format(self.initial)      
      for (s,a,d) in self.transitions:
        sym = a if a != EPSILON else "ε"
        res += "  \"{s}\" -> \"{d}\" [label = {a}];\n".format(s=s,d=d,a=sym)      
      res += "}"    
    output = Source(res)
    if outfilename:      
      output.render(outfilename)
    return output

##################
    
  def to_txtfile(self, outfilename:str=None) -> str:
    """
    Save automaton into txt file.
    """
    res = ""
    for (s,a,d) in self.transitions:
      res += "{} {} {}\n".format(s,a,d)          
    res += "A "
    res += " ".join([s for s in self.acceptstates])      
    if outfilename:
      if os.path.isfile(outfilename):
        warn("File {f} exists, will be overwritten",f=outfilename)
      with open(outfilename,"w") as outfile:
        print(res,file=outfile)
    return res
    
##################   

  def _repr_svg_(self):    
    return self.to_graphviz()._repr_svg_()

##################   

  def from_txt(self, source:str, name:str=None):
    """
    Reads from a txt source string and initializes automaton.
    """
    if self.statesdict :
      warn("Automaton {a} not empty: content will be lost",a=self.name)
    self.reset(name)
    rows = source.strip().split("\n")
    for (i,row) in enumerate(rows[:-1]):
      try:
        (src,symbol,dest) = row.strip().split(" ")
        self.add_transition(src,symbol,dest)
      except ValueError:
        error("Malformed triple {t}",pos=name+":"+str(i+1),t=row.strip())
    if not rows[-1].startswith("A"):
      error("File must end with \"A\" row",pos=name+":"+str(len(rows)))
    self.make_accept(rows[-1].strip().split(" ")[1:])

##################
    
  def from_txtfile(self, infilename:str):
    """
    Reads from txt file and initializes automaton.
    """    
    try:
      with open(infilename) as infile:
        rows = infile.readlines()
    except FileNotFoundError:
      error("File not found: {f}",f=infilename)
    name = os.path.splitext(os.path.basename(infilename))[0]
    return self.from_txt("".join(rows), name)
    
   
########################################################################
########################################################################

class RegExpReader(object):
  """
  A reader for regular expressions, mainly used to convert to postfix notation
  """
  
  exp:str
  hd:int
  
  def __init__(self,exp:str)->None:
    """
    re is an infix regular expression (union +, kleene * and concatenation)
    """
    self.exp = exp
    
  def to_postfix(self)->Optional[str]:
    """
    Convert current regexp to postfix notation using a top-down LL parser
    You'll learn about this parser in L3 - Compilation ;-)
    We implement the following context-free grammar (L2 Langages formels ;-)
    E -> C E'; E' -> '+' C E' | epsilon; C -> K C'; C' -> K C' | epsilon
    K -> '(' E ')' K' | a K'; K' -> '*' | epsilon
    """
    def re_error(fct:str, ex: str, found: str):
      error("\"{}\": \"{}\" expected, \"{}\" found".format(fct, ex, found))
    def elem(re:str)->bool:
      return re[self.hd].isalnum() or re[self.hd] == EPSILON
    def forward(re:str, ex: str):
      if re[self.hd] == ex : self.hd += 1
      else: re_error("forward",ex,re[self.hd])
    def e(re:str)->Optional[str]:
      if re[self.hd] == "(" or elem(re): return ebis(re, c(re))
      else: re_error("e","( or symbol", re[self.hd]); return None
    def ebis(re:str, h:str)->Optional[str]:
      if re[self.hd] in "+": forward(re, '+'); return h + ebis(re, c(re)) + "+"
      elif re[self.hd] in ")$": return h
      else: re_error("ebis",")+$ or symbol", re[self.hd]); return None
    def c(re:str)->Optional[str]:
      if re[self.hd] == "(" or elem(re): return cbis(re, k(re))
      else: re_error("c","( or symbol",re[self.hd]); return None
    def cbis(re:str,h:str)->Optional[str]:
      if re[self.hd] == "(" or elem(re): return h + cbis(re, k(re)) + "."
      elif re[self.hd] in "+)$": return h
      else: re_error("cbis","()+$ or symbol", re[self.hd]); return None
    def k(re:str)->Optional[str]:
      if elem(re) : r = re[self.hd]; forward(re, r)
      elif re[self.hd] == '(' : forward(re, '('); r = e(re); forward(re, ')')        
      else: re_error("k","( or symbol", re[self.hd]); return None
      return kbis(re,r)
    def kbis(re:str,h:str)->Optional[str]:
      if self.hd >= len(re): pdb.set_trace()
      if re[self.hd] == '*' : forward(re, '*'); return h + "*"
      elif re[self.hd] in "+()$" or elem(re): return h
      else: re_error("kbis","()+$* or symbol", re[self.hd]); return None
              
    self.hd = 0 # reading head, initialized once and kept during parsing
    result = e(self.exp+"$")
    if self.hd == len(self.exp):
      return result
    else: return error("Stopped at index {} \"{}\"".format(self.hd,self.exp[self.hd]))

########################################################################
########################################################################


########################################################################
########################################################################

class StackState(object):
  """
  Represents a state in the stack automaton, with its list of transitions
  """
  transitions: List[Tuple[str,str,str,Tuple[str],str]]
  name: str
  is_accept: bool
  
##################

  def __init__(self, name:str)->None:
    """
    Create a new state with a given name
    """
    self.name = name
    self.transitions = []  # by default, empty lis
    self.is_accept = False # by default, create non-accept state

##################
    
  def make_accept(self, accepts:bool=True):
    """
    Transform state into accept state, or the opposite, if accepts=False
    """
    self.is_accept = accepts

##################
   
  def add_transition(self, letter: str, head:str, push: List[str] , dest:'StackState'):
    """
    Add a transition on string 'letter', with stack head 'head' and push 'push' to StackState `dest`
    """
    trans = (letter, head, push, dest)
    if trans in self.transitions:
      warn("Redundant transition: {s} -{a},{h}/{p}-> {d}",s=self.name, a=letter, h=head, p=push, d=dest.name)
    else:
      self.transitions.append(trans)

##################

  def __str__(self)->str:
    """
    Standard function to obtain a string representation of a state
    """
    return self.name.replace('"',"&quot;")

########################################################################
########################################################################


class StackAutomaton(object):
  """
  A stack automaton is a list of stackstates and a pointer to the initial state
  Transitions and acceptance are represented inside states (see above)
  """  
  name:str
  statesdict:Dict[str,StackState]
  initial:StackState
  
##################  
  
  def __init__(self,name:str)->None:
    self.reset(name)     

##################

  def reset(self,name:str=None):
    """
    Reinitialize the automaton with empty content
    """
    if name:
      self.name = name
    elif not self.name:
      self.name = "NewStackAutomaton"
      warn("New stack automaton has no name")
    self.statesdict = OrderedDict()   
    self.initial = None 
    self.initial_stack = None

##################

  def deepcopy(self)->'StackAutomaton':
    """
    Make a deep copy of itself, that is a copy of all elements
    """
    a = StackAutomaton(self.name + "_copy")
    a.from_txt(self.to_txtfile())
    return a

##################
      
  def add_transition(self, src:str, letter:str, head:str, push:List[str], dst:str):
    """
    Add a transition from `src` to `dst` on `letter` with 'head' on the stack and pushing 'push'
    """    
    src_state = self.statesdict.get(src, StackState(src)) # create if absent    
    if not self.statesdict and not self.initial: 
      self.initial = src_state # by default first state added is initial
      self.initial_stack = head
    self.statesdict[src] = src_state # if new, add to dict
    dst_state = self.statesdict.get(dst, StackState(dst)) # create if absent
    self.statesdict[dst] = dst_state # if new, add to dict
    src_state.add_transition(letter, head, push, dst_state) # add the transition  
    

##################

  def remove_state(self, delstate:str):
    """
    Remove `delstate` from the automaton as well as all its transitions.
    """
    if delstate not in self.statesdict :     
      warn("State {} does not exist, will not remove".format(delstate))
      return    
    delstateobj = self.statesdict[delstate]
    del(self.statesdict[delstate]) # First, remove from main states list
    for state in self.statesdict.values(): # For all remaining states
      for (letter,head,push,dests) in state.transitions :
        if delstate== dests :
          state.transitions.remove((letter,head,push,dests))
          
##################

  def remove_transition(self, src:str, letter:str, head:str, push:List[str], dest:str):
    """
    Remove a transition from `src` to `dest` on `letter` with 'head' on the stack and pushing 'push'
    """ 
    if src not in self.statesdict :     
      warn("State {} does not exist, will not remove transition".format(src))
      return
    srcstate = self.statesdict[src]
    deststate = self.statesdict[dest]
    trans=(letter,head,push,deststate)
    if trans in srcstate.transitions:
      srcstate.transitions.remove(trans)
    else:
      warn("Transition {} -{},{}/{}-> {} not found".format(src,letter,head,push,dest))


##################

  @property
  def states(self) -> List[str]:
    return list(self.statesdict.keys())

##################

  @property
  def alphabet(self)->List[str]:
    """
    Get the set of symbols used in the current transitions (+ epsilon)
    """
    alphabet:Dict[str,None] = OrderedDict()
    for state in self.statesdict.values():
      for (s, head, push, dest) in state.transitions:       
        alphabet[s] = None
    return list(alphabet.keys())

##################

  @property
  def stack_alphabet(self)->List[str]:
    """
    Get the set of symbols used in the stack
    """
    alphabet:Dict[str,None] = OrderedDict()
    alphabet[self.initial_stack] = None
    for state in self.statesdict.values():
      for (letter, h, push, dest) in state.transitions:       
        alphabet[h] = None
        for p in push:
          alphabet[p] = None
    return list(alphabet.keys())


##################

  @property
  def acceptstates(self)->List[str]:
    """
    Return a set of accept states in the automaton.
    If the initial state is accepting, then it is the first in the list
    """
    accept = OrderedDict({k:None for (k,v) in self.statesdict.items() \
                                 if v.is_accept})
    if self.initial and self.initial.name in accept :
      result = [self.initial.name]
      del(accept[self.initial.name])
    else :
      result = []
    return result + list(accept.keys())
    
##################

  @property
  def transitions(self)->List[Tuple[str,str,str,list,str]]:
    """
    Returns a list of transitions, each represented as a 5-tuple.
    (source, letter, stack head, pushed symbols, destination)
    The first transition is always from the initial state with the initial stack symbol
    """
    result = []
    # Transitions from initial state 
    preresult=[]   
    for (letter, head, push, dest) in self.initial.transitions:
      if head == self.initial_stack:
        result.append((self.initial.name, letter, head, push, dest.name))
      else:
        preresult.append((self.initial.name, letter, head, push, dest.name))
    result+=preresult       
    # Transitions from other states      
    for source in self.statesdict.values():
      if source != self.initial :
        for (letter, head, push, dest) in source.transitions:
          result.append((source.name, letter, head, push, dest.name))
    return result
          

##################

  def rename_state(self, oldname: str, newname: str):
    """
    Renames a state in the automaton from `oldname` to `newname`
    """
    if newname in self.states :
      warn("New name \"{}\" already exists, try a new one".format(newname))
      return
    try :
      self.statesdict[newname] = self.statesdict[oldname]
      self.statesdict[newname].name = newname
      del self.statesdict[oldname]
    except KeyError:
      warn("Tried to rename not-existent state \"{}\"".format(oldname))

	  
##################
    
  def make_accept(self, src:Union[str,List[str]], accepts:bool=True):
    """
    Transform the a state(s) of the automaton into accept state(s)
    """
    if isinstance(src,str):
      src = [src] # transform in list if necessary
    for srci in src:
      if srci not in self.statesdict:
        error("Accept state {a} inexistent!",a=srci)
      self.statesdict[srci].make_accept(accepts)
    
##################

  def __str__(self)->str:
    """
    Standard function to obtain a string representation of an automaton
    """
    
    dlist=""
    for (src,letter,head,push,dest) in self.transitions:
       p= ".".join(push) if push != [] else "ε"
       l= letter if letter !="%" else "ε"
       dlist+= src+" --"+l+","+head+"/"+ p + "--> "+dest+"\n"
    tpl = "{A} = <Q={{{Q}}}, S={{{S}}}, Z={{{Z}}}, D, Z0={Z0}, q0={q0}, F={{{F}}}>\nD =\n{D}"    
    return tpl.format(A=self.name, Q=str(",".join(self.states)),
                      S=",".join(map(lambda x: x if x != "%" else "ε",self.alphabet)),
                      Z=",".join(self.stack_alphabet), q0=self.initial, Z0=self.initial_stack,
                      F=",".join(self.acceptstates),
                      D=dlist)

##################
    
  def to_graphviz(self, outfilename:str=None) -> Source:
    if not self.states:
      tpl = "digraph L{{label=\"{name}\"; node [shape=record]; a [label=\"empty\"]}}"
      res = tpl.format(name=self.name)
    else:
      res = """digraph finite_state_machine {
  rankdir=LR;  
  size=\"8,5\""""
      res += "  label=\"{}\"".format(self.name)
      if self.acceptstates:
        accept = " ".join(map(lambda x : '"'+x+'"', self.acceptstates))
        res += "  node [shape = doublecircle]; {};\n".format(accept)   
      res += "  node [shape = circle];\n"
      res += "  __I__ [label=\"\", style=invis, width=0]\n"
      res += "  __I__ -> \"{}\"\n".format(self.initial)      
      for (source,letter, head, push, dest) in self.transitions:
        l=""
        for s in push:
          l+=s
        sym = letter if letter != EPSILON else "ε"
        lsym = l if l != "" else "ε"
        res += "  \"{source}\" -> \"{dest}\" [label = \"{sym},{head}->{l}\"];\n".format(source=source,dest=dest,sym=sym, head=head,l=lsym)      
      res += "}"    
    output = Source(res)
    if outfilename:      
      output.render(outfilename)
    return output

##################
    
  def to_txtfile(self, outfilename:str=None) -> str:
    """
    Save automaton into txt file.
    """
    res = ""
    for (source, a, head, push, dest) in self.transitions:
      l=""
      linit= True
      for s in push:
        if linit:
          linit=False
        else:
          l+="."  
        l+=s
      lsym = l if l != "" else "%"
      res += "{} {} {} {} {}\n".format(source, a, head, lsym, dest)          
    res += "A "
    res += " ".join([s for s in self.acceptstates])      
    if outfilename:
      if os.path.isfile(outfilename):
        warn("File {f} exists, will be overwritten",f=outfilename)
      with open(outfilename,"w") as outfile:
        print(res,file=outfile)
    return res
    
##################   

  def _repr_svg_(self):    
    return self.to_graphviz()._repr_svg_()

##################   

  def from_txt(self, source:str, name:str=None):
    """
    Reads from a txt source string and initializes automaton.
    """
    if self.statesdict :
      warn("Automaton {a} not empty: content will be lost",a=self.name)
    self.reset(name)
    rows = source.strip().split("\n")
    for (i,row) in enumerate(rows[:-1]):
      try:
        (src,letter, head, pointpush ,dest) = row.strip().split(" ")
        push = [] if pointpush =="%" else pointpush.strip().split(".")
        self.add_transition(src,letter, head,push,dest)
      except ValueError:
        error("Malformed tuple {t}",pos=name+":"+str(i+1),t=row.strip())
    if not rows[-1].startswith("A"):
      error("File must end with \"A\" row",pos=name+":"+str(len(rows)))
    self.make_accept(rows[-1].strip().split(" ")[1:])

##################
    
  def from_txtfile(self, infilename:str):
    """
    Reads from txt file and initializes automaton.
    """    
    try:
      with open(infilename) as infile:
        rows = infile.readlines()
    except FileNotFoundError:
      error("File not found: {f}",f=infilename)
    name = os.path.splitext(os.path.basename(infilename))[0]
    return self.from_txt("".join(rows), name)


##################
  def rename_stack_alphabet(self):
    i=0
    rem=[]
    add=[]
    for x in self.stack_alphabet:
      print(x)
      while "X"+str(i) in self.stack_alphabet:
        i+=1
      new_x= "X"+str(i)
      for (src, letter, head, push, dest) in self.transitions:
        print(push)
        if x in push:
          rem.append((src, letter, head, push, dest))
          add.append((src, letter, head if head !=x else new_x , map(lambda y: y if y !=x else new_x, push), dest))
          print(self)



##################
  def push_reduce(self):
    for (src,letter,head,push,dest) in self.transitions:
      if len(push) >2:
        self.remove_transition(src,letter,head,push,dest)
        h=head
        hh=push[0]
        d=src+letter+head+"".join(push)+dest
        for i in range(len(push)-1,1,-1):
          hh=d+str(i)
          hhh=push[i]
          self.add_transition(src,"%",h,[hh,hhh],src)
          h=hh
        self.add_transition(src,letter,h,[hh,push[0]],dest)
    print(self)
    self.rename_stack_alphabet()

   
########################################################################
########################################################################


class Variable(object):
  """
  Represents a non terminal in a grammar with its list of rules
  """
  rules: List[List[str]]
  name: str
  
##################

  def __init__(self, name:str)->None:
    """
    Create a variable with a given name
    """
    self.name = name
    self.rules = []  # by default, empty list

##################
   
  def add_rule(self, prod: List[str]):
    """
    Add a rule producing 'prod'
    """
    if prod in self.rules:
      warn("Redundant rule: {x} --> {p}",x=self.name, p=prod)
    else:
      self.rules.append(prod)

##################
  def remove_rule(self, prod:List[str]):
    """
    Remove a rule producting 'prod' from the variable
    """ 
    if prod not in self.rules :     
      warn("rule {} -> {} does not exist, will not remove".format(self.name, prod))
      return
    else:
      self.rules.remove(prod)

##################
  def __str__(self)->str:
    """
    Standard function to obtain a string representation of a variable
    """
    return self.name.replace('"',"&quot;")
########################################################################
########################################################################

class Grammar(object):
  """
  A grammar is a dictionary of variables and a pointer to the initial variable.
  Rule are represented inside variables (see above)
  """  
  name:str
  variablesdict:Dict[str,Variable]
  initial:Variable
  
##################  
  
  def __init__(self,name:str)->None:
    self.reset(name)     

##################

  def reset(self,name:str=None):
    """
    Reinitialize the automaton with empty content
    """
    if name:
      self.name = name
    elif not self.name:
      self.name = "NewGrammar"
      warn("New grammar has no name")
    self.variablesdict = OrderedDict()   
    self.initial = None

##################

  def deepcopy(self)->'Grammar':
    """
    Make a deep copy of itself, that is a copy of all elements
    """
    a = Grammar(self.name + "_copy")
    a.from_txt(self.to_txtfile())
    return a

##################
      
  def add_rule(self, var:str, prod:List[str]):
    """
    Add a rule 'var' -> 'prod'
    """    
    src_var = self.variablesdict.get(var, Variable(var)) # create if absent    
    if not self.variablesdict and not self.initial: 
      self.initial = src_var # by default first variable added is initial
    self.variablesdict[var] = src_var # if new, add to dict
    src_var.add_rule(prod) # add the transition  
    

##################

  def remove_var(self, delvar:str):
    """
    Remove `delvar` from the grammar.
    """
    if delvar not in self.variablesdict :     
      warn("Variable {} does not exist, will not remove".format(delvar))
      return    
    del(self.variablesdict[delvar]) # remove from variable dict

          
##################

  def remove_rule(self, src:str, rule:List[str]):
    """
    Remove a rule 'src' -> 'rule'
    """ 
    if src not in self.variablesdict :     
      warn("Variable {} does not exist, will not remove".format(src))
      return    
    self.variablesdict[src].remove_rule(rule) # remove rule from variable


##################

  @property
  def variables(self) -> List[str]:
    return list(self.variablesdict.keys())

##################

  @property
  def alphabet(self)->List[str]:
    """
    Get the set of symbols which are not variables
    """
    alphabet:Dict[str,None] = OrderedDict()
    for var in self.variablesdict.values():
      for prod in var.rules: 
        for s in prod:
          if not s in self.variables:      
            alphabet[s] = None
    return list(alphabet.keys())
    
##################

  @property
  def rules(self)->List[Tuple[str,List[str]]]:
    """
    Returns a list of rules, each represented as a pair.
    (source, production)
    The first rule is always from the initial variable
    """
    result = []
    # rules from initial variable
    for prod in self.initial.rules:
      result.append((self.initial.name, prod))
    for var in self.variablesdict.values():
      if var != self.initial:
        for prod in var.rules:
          result.append((var.name,prod))
    return result
          

##################

  def rename_var(self, oldname: str, newname: str):
    """
    Renames a variable in the grammar from `oldname` to `newname`
    """
    if newname in self.variables :
      warn("New name \"{}\" already exists, try a new one".format(newname))
      return
    try :
      self.variablesdict[newname] = self.variablesdict[oldname]
      self.variablesdict[newname].name = newname
      del self.variablesdict[oldname]
      for var in self.variablesdict.values():
        for prod in var.rules:
          if oldname in prod:
            var.add_rule(map(lambda x: x if x != oldname else newname))
            var.remove_rule(prod)
          
    except KeyError:
      warn("Tried to rename not-existent variable \"{}\"".format(oldname))
    
##################

  def __str__(self)->str:
    """
    Standard function to obtain a string representation of a grammar
    """
    
    plist=""
    for src in self.variablesdict.values():
      plist+= src.name
      if src.rules != []:
        plist+= " --> " + " | ".join(map(lambda x: ".".join(x),map(lambda x: x if x != [] else ["ε"],src.rules)))
      plist+="\n"
    tpl = "{G} = <A={{{A}}}, N={{{N}}}, P, S0={S0}>\nP =\n{P}"    
    return tpl.format(G=self.name,
                      A=",".join(self.alphabet),
                      N=",".join(self.variables), S0=self.initial,
                      P=plist)

##################
    
  def to_txtfile(self, outfilename:str=None) -> str:
    """
    Save grammar into txt file. One rule per line. The first rule written is a rule from the initial variable
    """
    res = ""
    for (src, prod) in self.rules:
      res+= src+" "
      if prod != []:
        res+= ".".join(prod)
      else:
        res+= "%"
      res+= "\n"  
    if outfilename:
      if os.path.isfile(outfilename):
        warn("File {f} exists, will be overwritten",f=outfilename)
      with open(outfilename,"w") as outfile:
        print(res,file=outfile)
    return res

##################   

  def from_txt(self, source:str, name:str=None):
    """
    Reads from a txt source string and initializes grammar. The first line determines the initial variable.
    """
    if self.variablesdict :
      warn("Grammar {a} not empty: content will be lost",a=self.name)
    self.reset(name)
    rows = source.strip().split("\n")
    for (i,row) in enumerate(rows):
      try:
        (src,prod) = row.strip().split(" ")
        realprod= [] if prod =="%" else prod.strip().split(".")
        self.add_rule(src,realprod)
      except ValueError:
        error("Malformed tuple {t}",pos=name+":"+str(i+1),t=row.strip())

##################
    
  def from_txtfile(self, infilename:str):
    """
    Reads from txt file and initializes grammar.
    """    
    try:
      with open(infilename) as infile:
        rows = infile.readlines()
    except FileNotFoundError:
      error("File not found: {f}",f=infilename)
    name = os.path.splitext(os.path.basename(infilename))[0]
    return self.from_txt("".join(rows), name)


    
   
########################################################################
########################################################################

if __name__ == "__main__": # If the module is run from command line, test it
  
  a= StackAutomaton("aut1")
  a.add_transition("0","a","X",["Y","Z","Z"],"1")
  a.add_transition("1","b","X",[],"1")
  a.make_accept("1")
  a.to_txtfile("test/aut1.ap")
  b= StackAutomaton("aut2")
  b.from_txtfile("test/anbn.ap")
  print(b)
  print(a)
 
  G=Grammar("g_test")
  G.add_rule("S",["a","S","b"])
  G.add_rule("S",[])
  print(G)
  
  


    
