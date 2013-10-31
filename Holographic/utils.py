# Python Holographic Reduced Representation Library
# -------------------------------------------------
# (C) Copyright Andrew Mundy 2013

from numpy import *
import numbers

def vec_generate( d ):
	"""Generates a vector of dimensionality d, with elements selected from
	a normal distribution with mean 0 and variance 1/d."""
	return random.normal( 0, sqrt(1./d), size=(d) )

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

def vec_exponentiate( a, n ):
	"""Raise vector a to the power of n."""
	assert isinstance( a, array ) and isinstance( n, numbers.Number )

        # Transform into the Fourier/frequency domain and perform
        # element-wise multiplication.
        fft_a = fft.fft( a )
        fft_b = fft_a ** n

        # Now convert back from the Fourier domain
        return real( fft.ifft( fft_b ) )
