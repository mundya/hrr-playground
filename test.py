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
a = mem.add_symbol( 'a' )	# Create a symbol "a"
b = mem.add_symbol( 'b' )	# Create a symbol "b"

# Now generate the inverses of these symbols
sys.stderr.write( "\n > Inverting the initial vectors..." )
a_ = vec_invert( a )	# a' is the rough inverse of a
b_ = vec_invert( b )	# b' is the rough inverse of b

# Convolve a and b to generate c
sys.stderr.write( "\n > Convolving..." )
sys.stderr.write( "\n   > c := a (*) b" )
c = vec_convolve_circular( a, b )	# c := a (*) b

# Convolve c with a' to get ca (~= b)
sys.stderr.write( "\n   > a' (*) c" )
ca = vec_convolve_circular( a_, c )

# Convolve c with b' to get cb (~= a)
sys.stderr.write( "\n   > b' (*) c" )
cb = vec_convolve_circular( b_, c )

sys.stderr.write( "\n   > b' (*) c" )

# Use the clean up memory to illustrate similarities
for (l, s) in [ ( 'a', a ),
		( 'b', b ),
		( "a'", a_ ),
		( "b'", b_ ),
		( "a(*)b", c ),
		( "(a(*)b) (*) a' ~= b", ca ),
		( "(a(*)b) (*) b' ~= a", cb ),
	]:
	sys.stdout.write( "\n%s is similar to: \n" % l )
	print( mem.cleanup( s ) )

sys.stdout.write( "\n" )
