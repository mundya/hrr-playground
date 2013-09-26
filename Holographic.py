from random import normalvariate
import math

def vec_conform( f ):
	"""Decorator to ensure that vectors passed as arguments are of the same dimensionality."""
	def f_( a, b ):
		if len( a ) != len( b ):
			raise Exception( "Vectors are not of the same dimensionality." )

		return f(a, b)
	return f_
		
@vec_conform
def vec_convolve_circular( a, b ):
	"""Performs circular convolution on two vectors of the same length."""
	# Could do this with DFT/FFT if preferred	
	# c = a (*) b
	n = len( a )
	return [ sum( [ a[k] * b[ (j-k) % n ] for k in range( n ) ] ) for j in range( n ) ]

def vec_invert( a ):
	"""Perform a rough inversion of the given vector."""
	return [a[0]] + [ a[i] for i in range( len(a) - 1, 0, -1 ) ]

def vec_create( n ):
	"""Create a vector of the given dimensionality (n) such that the elements
	are selected to lie on a normal distribution of mean 0 and variance 1/n."""
	mu = 0
	sigma = math.sqrt( 1.0/n )	# (Plate 1991)
	return [ normalvariate( mu, sigma ) for i in range( n ) ]

def vec_magnitude( a ):
	return math.sqrt( sum( [ x**2 for x in a ] ) )

@vec_conform
def vec_dot_product( a, b ):
	"""Returns dot product of two vectors of the same dimensionality."""
	n = len( a )
	return sum( [ a[i]*b[i] for i in range( n ) ] )

@vec_conform
def vec_angle( a, b ):
	"""Return the angle between the two vectors."""
	return math.acos( vec_dot_product( a, b ) / ( vec_magnitude(a) + vec_magnitude(b) ) )

class CleanUpMemory( object ):
	"""Creates a clean up memory ... <MORE HERE> ..."""
	def __init__( self, dimensionality ):
		self._memory = {}
		self._dimensionality = dimensionality
	
	def add_symbol( self, label ):
		"""Generate a new vector and add it to the memory with the given
		label."""
		self._memory[label] = vec_create( self._dimensionality )
		return self._memory[label]
	
	def cleanup( self, a ):
		"""Clean up the given vector."""
		matches = sorted(
				[ (vec_dot_product( a, b ) / ( vec_magnitude(a) * vec_magnitude(b) ), l) for (l,b) in self._memory.items() ],
				key = lambda (x,y) : x
			  )
		matches.reverse()
		return matches
