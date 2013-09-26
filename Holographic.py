"""
Holographic Reduced Representations
===================================
Andrew Mundy

Uses NUMPY to construct symbol representations with Holographic Reduced
Representations.  HRR vectors may be convolved and cleaned up using any given
clean-up memory.
"""

from numpy import *

def vec_convolve_circular( a, b ):
	"""Convolve two vectors and return the result."""
	# Check that the vectors conform
	if not a.size == b.size:
		raise ValueError( "Vectors must be of the same dimensionality." )

	# Transform into the Fourier/frequency domain and perform
	# element-wise multiplication.
	fft_a = fft.fft( a )
	fft_b = fft.fft( b )
	fft_c = fft_a * fft_b

	# Now convert back from the Fourier domain
	return real( fft.ifft( fft_c ) )

class Symbol( object ):
	"""A symbol object represents a symbol within an HRR system.  For ease
	of use a label is stored with the symbol, and the entire construct is
	immutable once generated."""

	def __init__( self, label, **kwargs ):
		"""Generate a new symbol with the given label and
		dimensionality.  The label is used only to make sense of what's
		going on. If the vector for the symbol is specified this is
		used, otherwise a correctly distributed vector is generated.

		A generator may be specified to allow for custom vector
		generation.  The generator must accept one parameter which is
		the dimensionality of the desired vector and must return a numpy
		array of the correct size.

		The default generator would be passed as:
		generator = lambda d : random.normal( 0, sqrt(1./d), size=d )
		"""
		# Store the constants
		self._label = label

		# Check that either a dimensionality or a vector is passed
		# as a parameter.
		if not "vector" in kwargs and not "dimensionality" in kwargs:
			raise ValueError( "You must either specify a vector"\
					  "or a dimensionality." )

		# See if a generator has been passed, otherwise use the default
		if not "generator" in kwargs:
			gen = lambda d : random.normal( 0, sqrt(1./d), d )
		else:
			gen = kwargs["generator"]

		# Generate and store the vector representing this symbol
		if not "vector" in kwargs:
			self._d = kwargs["dimensionality"]
			self._v = gen( self._d )
		else:
			self._v = array( kwargs["vector"] )
			self._d = self._v.size
	
	# Getter methods
	def __str__( self ):
		return self._label
	
	def dimensionality( self ):
		return self._d
	
	def vector( self ):
		return self._v

	def inverse_vector( self ):
		"""Return a rough inverse of the vector for the symbol."""
		return hstack( [ self._v[0], self._v[-1:0:-1] ] )
	
	# Bind and unbind operations
	def bind( self, b ):
		"""Bind the current symbol to the given symbol and return
		a new vector with this binding."""
		# Convolve to form the combined vector
		c = vec_convolve_circular( self.vector(), b.vector() )

		# Generate a new label
		l = "(%s (*) %s)" % ( self, b )

		# Return a new symbol
		return Symbol( l, vector = c )
	
	def unbind( self, b ):
		"""Unbinds the given vector (b) from the current vector."""
		# Convolve to form the combined vector
		c = vec_convolve_circular( self.vector(), b.inverse_vector() )

		# Generate a new label
		l = "(%s (*) %s')" % ( self, b )

		# Return a new symbol
		return Symbol( l, vector = c )
	
	# Combine operations
	def add( self, b ):
		"""Add the given vector to the current vector."""
		pass
	
	# Equivalence / comparison operations
	def dot_product( self, b ):
		"""Returns the dot product of the given symbols."""
		return sum( self.vector() * b.vector() )
	
	def comparison( self, b ):
		"""Returns the cosine of the angle between the given symbols."""
		return self.dot_product( b ) / ( self.magnitude() * b.magnitude() )
	
	# Other useful bits
	def magnitude( self ):
		"""Returns the magnitude of the vector representing the
		symbol."""
		return sqrt( sum( self.vector() ** 2 ) )

class CleanUpMemory( object ):
	"""A CleanUpMemory provides a source of Symbols and is capable of
	cleaning up bound and unbound symbols."""

	def __init__( self, dimensionality ):
		"""Create a new CleanUpMemory with the given dimensionality.
		All symbols derived from this memory will have this
		dimensionality."""
		self._symbols = []
		self._dimensionality = dimensionality
	
	def get_symbol( self, label ):
		"""Instantiate a new symbol with the given label, the symbol
		will be added to the CleanUpMemory."""
		# Generate the new symbol, store it and then return the symbol
		# to the caller.
		s = Symbol( label, dimensionality = self._dimensionality )
		self._symbols.append( s )
		return s
	
	def clean_up( self, s ):
		"""Return an ordered list of stored symbols along with their
		similarity to the provided symbol."""
		return sorted(
				[ (s_, s.comparison( s_ )) for s_ in self._symbols],
				key = lambda (s_, c) : c,
				reverse = True
		)
