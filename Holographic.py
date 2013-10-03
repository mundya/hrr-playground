"""
Holographic Reduced Representations
===================================
Andrew Mundy

Uses NUMPY to construct symbol representations with Holographic Reduced
Representations.  HRR vectors may be convolved and cleaned up using any given
clean-up memory.
"""

from numpy import *
import operator
from numbers import Number

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

def vec_generate( d ):
	"""Generates a vector of dimensionality d, with elements selected from
	a normal distribution with mean 0 and variance 1/d."""
	return random.normal( 0, sqrt(1./d), size=(d) )

def vec_generate_normalised( d ):
	"""Generates a normalised vector of dimensionality d, whose elements
	are initially selected from a normal distribution of mean 0 and
	variance 1/d."""
	return vec_normalise( vec_generate( d ) )

def vec_normalise( v ):
	"""Returns a normalised copy of the vector."""
	v = array( v )
	m = sqrt( sum( v ** 2 ) )
	return v / m

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
		generator = lambda d : random.normal( 0, sqrt(1./d), size=(d) )
		"""
		# Store the constants
		self._label = label

		# Check that either a dimensionality or a vector is passed
		# as a parameter.
		if not "vector" in kwargs and not "dimensionality" in kwargs:
			raise ValueError( "You must either specify a vector "\
					  "or a dimensionality." )

		# See if a generator has been passed, otherwise use the default
		if not "generator" in kwargs:
			gen = vec_generate
		else:
			gen = kwargs["generator"]

		# Generate and store the vector representing this symbol
		if not "vector" in kwargs:
			self._d = kwargs["dimensionality"]
			self._v = gen( self._d )

			if not self._v.size == self._d:
				raise ValueError( "Generated vector is not of "\
						  "the correct dimensionality." )
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
	def _add( self, b, op = operator.add ):
		"""Generic add or subtract for vectors."""
		# Check they conform
		if not self.vector().size == b.vector().size:
			raise ValueError( "Vectors must be of the same "\
					  "dimensionality." )

		# Get the operator character
		op_cs = { operator.add: '+', operator.sub: '-' }
		op_c = op_cs[ op ]

		# Generate the new label
		l = "( %s %c %s )" % ( self, op_c, b )

		# Generate the new vector
		c = op( self.vector(), b.vector() )

		# Return the new symbol
		return Symbol( l, vector = c )

	def __add__( self, b ):
		"""Add the given vector to the current vector."""
		return self._add( b )
	
	def __sub__( self, b ):
		"""Subtract the given vector from the current vector."""
		return self._add( b, op = operator.sub )
	
	# Multiplication
	def __mul__( self, k ):
		"""Return a scaled version of the symbol, multiplied by a
		scalar value."""
		if not isinstance( k, Number ) and not isinstance( k, Symbol ):
			raise ValueError( "Symbols may only be scaled by"\
					  "scalar values or bound with other "\
					  "symbols." )

		if isinstance( k, Symbol ):
			# Bind instead of scale
			return self.bind( k )

		# Generate a new label
		l = "%.3f%s" % (k, self )

		# Generate the new vector
		c = k * self.vector()

		# Return the new symbol
		return Symbol( l, vector = c )
		
	__rmul__ = __mul__ # Allows 2*x as well as x*2
	
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

	def __init__( self, dimensionality, **kwargs ):
		"""Create a new CleanUpMemory with the given dimensionality.
		All symbols derived from this memory will have this
		dimensionality.
		A custom generator may be specified using the rules given in
		the specification for the Symbol class, this is passed as a
		keyword argument, "generator = ..."
		"""
		self._symbols = []
		self._dimensionality = dimensionality
		self._generator = None

		# Generate a null vector
		self._null = Symbol( "null%d" % dimensionality,
				     vector = zeros( dimensionality ) )
		self._symbols.append( self._null )

		if "generator" in kwargs:
			self._generator = kwargs["generator"]
	
	def add_symbol( self, symbol ):
		"""Add a given symbol to the memory."""
		if not symbol.vector().size == self._dimensionality:
			raise ValueError( "Symbol must have the same dimensionality as the memory." )

		# Add the symbol to the memory
		self._symbols.append( symbol )
	
	def get_symbol( self, label ):
		"""Instantiate a new symbol with the given label, the symbol
		will be added to the CleanUpMemory."""
		# Generate the new symbol, store it and then return the symbol
		# to the caller.
		if self._generator is None:
			s = Symbol( label, dimensionality = self._dimensionality )
		else:
			s = Symbol( 
					label,
					dimensionality = self._dimensionality,
					generator = self._generator
			)
		self.add_symbol( s )
		return s
	
	def null( self ):
		"""Returns the null symbol for the memory."""
		return self._null
	
	def clean_up( self, s ):
		"""Return an ordered list of stored symbols along with their
		similarity to the provided symbol."""
		return sorted(
				[ (s_, s.comparison( s_ )) for s_ in self._symbols],
				key = lambda (s_, c) : c,
				reverse = True
		)
