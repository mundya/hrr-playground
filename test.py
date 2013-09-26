#!/usr/bin/python
"""
Holographic Memory Test / Example
=================================
Andrew Mundy

Generates a clean up memory, instantiates some symbols, illustrates
convolving and de-convolving these symbols before cleaning up the
output.
The dimensionality of the vectors is passed as a command line argument.
"""

# Import the library
from Holographic import *
import sys

# Check that the dimensionality has been specified
if len( sys.argv ) < 2:
	sys.stderr.write( "Usage: test.py <dimensionality>\n" )
	sys.exit( 1 )

# Save the dimensionality
d = int( sys.argv[1] )
sys.stdout.write( "Testing the Holographic Reduced Representation Library and Clean Up Memory\n" )
sys.stdout.write( "Dimensionality = %d\n" % d )

# Create the clean up memory, and generate two symbols
sys.stderr.write( "\n > Generating the clean up memory and vectors..." )
mem = CleanUpMemory( d )
[a,b] = map( mem.get_symbol, ['a', 'b' ] )	# Create symbols 'a' and 'b'

# Bind a to b to get c
sys.stderr.write( "\n > Binding and unbinding..." )
sys.stderr.write( "\n   > c := a (*) b" )
c = a.bind( b )

# Unbind a and b from c respectively
sys.stderr.write( "\n   > a (*) b (*) a' ~= b" )
ca_ = c.unbind( a )

sys.stderr.write( "\n   > a (*) b (*) b' ~= a" )
cb_ = c.unbind( b )

# Illustrate the similarities with items stored in the memory
sys.stderr.write( "\n > Illustrating similarities...\n" )
for s in [a,b,c,ca_,cb_]:
	sys.stdout.write( "\n%s is similar to:" % s )

	for (s_, c_) in mem.clean_up( s ):
		sys.stdout.write( "\n\t%s -> %.3f" % (s_, c_) )
	
	sys.stdout.write( "\n" )
