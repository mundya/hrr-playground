# Python Holographic Reduced Representation Library
# -------------------------------------------------
# (C) Copyright Andrew Mundy 2013

from Symbol import *

def parent_match( f ):
    def f_( self, symbol ):
        if not self.factory == symbol.parent:
            raise ValueError( "You may only use Symbols with " \
                              "CleanUpMemories which share a " \
                              "SymbolFactory." )

        return f( self, symbol )

    return f_

class CleanUpMemory( object ):
    """A CleanUpMemory acts to clean up noisy versions of symbols.  All
    symbols used with the memory must be drawn from the same factory."""

    def __init__( self, factory ):
        """Create a new CleanUpMemory with the given SymbolFactory."""
        # Save the factory and create an empty list of symbols
        self.factory = factory
        self.symbols = []
    
    @parent_match
    def add_symbol( self, symbol ):
        """Add the given symbol to the memory."""
        self.symbols.append( symbol )
    
    @parent_match
    def clean( self, symbol ):
        """Return a list of symbols along with their similarity to the
        given symbol."""
        comparisons = map( lambda s : ( symbol.compare( s ), s ),
	                   self.symbols )
        comparisons.sort( key = lambda ( c, s ) : c, reverse = True )
        return comparisons
    
    @parent_match
    def cleanest( self, symbol ):
        """Return the cleanest version of symbol in this memory."""
        return self.clean( symbol )[0][1]
