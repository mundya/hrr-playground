# Python Holographic Reduced Representation Library
# -------------------------------------------------
# (C) Copyright Andrew Mundy 2013

from utils import *

def parent_match( f ):
	def f_( self, other ):
		if not self.parent == other.parent:
			raise ValueError( "Only Symbols drawn from the same " \
					  "SymbolFactory may be used in this operation." )

		return f( self, other )
	
	return f_

class SymbolFactory( object ):
	"""A SymbolFactory provides a common source of symbols and vectors."""
	def __init__( self, dimensionality, vec_generator, symbol_type ):
		"""Create a new SymbolFactory which will produce symbols of
		the type symbol_type, using the generator vec_generator and
		with the given dimensionality.
		The generator must accept a dimensionality and return a vector
		of that size.
		The Symbol type can probably be a function which accepts a reference
		to this factory, a label and a vector and returns an appropriate type
		of Symbol."""
		# Store the constants
		self.dimensionality = lambda : dimensionality
		self.generate_vector = lambda : vec_generator( self.dimensionality() )
		self._symbol_type = symbol_type
	
	def make_symbol( self, label, vector ):
		"""Make a new symbol with the given label and vector."""
		vector = array( vector )
		
		if not vector.size == self.dimensionality():
			raise ValueError( "Can only make Symbols with dimensionality " \
					  "%d" % self.dimensionality() )

		s = self._symbol_type( self, label, vector )
		assert isinstance( s, Symbol )
		return s
	
	def new_symbol( self, label ):
		"""Generate a new symbol with the given label."""
		# Create a vector
		v = self.generate_vector()

		# Create and return the symbol
		return self.make_symbol( label, v )

class Symbol( object ):
	"""Provides the interface expected of all Symbols.  Typically
	a derived Symbol type will be used (e.g. SaturatingSymbol).

	A symbol object represents a symbol within an HRR system.  For ease
	of use a label is stored with the symbol, and the entire construct is
	immutable once generated."""

	def __init__( self, parent, label, vector ):
		"""Generate a new symbol with the given label and
		vector.  The label is used only to make sense of what's
		going on.
		The parent represents the SymbolFactory that the Symbol is drawn
		from. Only Symbols from the same Factory may be bound or combined.
		"""
		# Store the constants as the return values of functions
		self.parent = parent
		self.label = lambda : label
		self.vector = lambda : array( vector )
		self.dimensionality = lambda : self.vector().size
	
	def __str__( self ):
		return self.label()
	
	def inverse( self ):
		"""Returns the approximate inverse of the symbol for use in
		unbinding operations."""
		# Create the inverse vector
		v = hstack( [ self.vector()[0], self.vector()[-1:0:-1] ] )

		# Create a new Symbol of this type with label' and the vector
		return self.parent.make_symbol( "%s'" % self, v )
	
	@parent_match
	def bind( self, other ):
		"""Bind two Symbols of equivalent type and dimensionality."""
		# Combine the labels
		l_ = "( %s (*) %s )" % (self, other)

		# Combine the vectors
		v_ = vec_convolve_circular( self.vector(), other.vector() )

		# Create and return a new Symbol
		return self.parent.make_symbol( l_, v_ )
	
	@parent_match
	def unbind( self, other ):
		"""Unbind two Symbols of equivalent type and dimensionality."""
		return self.bind( other.inverse() )
	
	@parent_match
	def compose( self, other ):
		"""Create the composition of two Symbols of equivalent type and
		dimensionality."""
		# Combine the labels
		l_ = "( %s + %s )" % ( self, other )

		# Combine the vectors
		v_ = self.vector() + other.vector()

		# Create and return a new Symbol
		return self.parent.make_symbol( l_, v_ )
	
	def scale( self, scale ):
		"""Return the current Symbol scaled by some factor."""
		if not isinstance( scale, numbers.Number ):
			raise ValueError( "You may only scale a Symbol by a Number." )

		# Create the new label
		l_ = "( %.3f %s )" % ( scale, self )

		# Generate the new vector
		v_ = scale * self.vector()

		# Create and return a new Symbol
		return self.parent.make_symbol( l_, v_ )
	
	def exponentiate( self, n ):
		"""Return the current Symbol raised to the power of n."""
		if not isinstance( n, numbers.Number ):
			raise ValueError( "You may only scale a exponentiate by a Number." )

		# Create the new label
		l_ = "( %s^{%.3f} )" % ( self, n )

		# Generate the new vector
		v_ = vec_exponentiate( self.vector(), n )

		# Create and return a new Symbol
		return self.parent.make_symbol( l_, v_ )
	
	@parent_match
	def compare( self, other ):
		"""Compare this Symbol with another."""
		return vec_cosine( self.vector(), other.vector() )
	
	def magnitude( self ):
		"""Return the magnitude of the vector representing this Symbol."""
		return vec_magnitude( self.vector() )

class SaturatingSymbol( Symbol ):
	"""A Symbol which models saturation of the value of components."""

	def __init__( self, parent, label, vector, saturation = lambda x : x ):
		"""Create a new SaturatingSymbol, the saturation function 
		(default is the identity function) will be applied to each
		component of the initial vector when instantiating the vector."""
		# Saturate the input vector
		vector = saturation( vector )

		# Now act as normal by calling super init
		super( SaturatingSymbol, self ).__init__( parent, label, vector )
