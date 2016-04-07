#!/usr/bin/env python

# A simple python program that reads lines from stdin, makes a simple alteration, 
# and sends the result back to stdout
import sys
from string import strip
for line in sys.stdin.readlines():
    line=strip(line)
    print 'This Is '+line