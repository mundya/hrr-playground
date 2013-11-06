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
    assert isinstance( a, ndarray ) and isinstance( b, ndarray )
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
    assert isinstance( a, ndarray ) and isinstance( n, numbers.Number )

    # Transform into the Fourier/frequency domain and perform
    # element-wise multiplication.
    fft_a = fft.fft( a )
    fft_b = fft_a ** n

    # Now convert back from the Fourier domain
    return real( fft.ifft( fft_b ) )

def vec_magnitude( a ):
    """Return the magnitude of vector a."""
    assert isinstance( a, ndarray )
    return sqrt( sum( a**2 ) )

def vec_dot_product( a, b ):
    """The dot product of two vectors."""
    assert isinstance( a, ndarray ) and isinstance( b, ndarray )
    return sum( a * b )

def vec_cosine( a, b ):
    """Return the cosine of the angle between the vectors."""
    assert isinstance( a, ndarray ) and isinstance( b, ndarray )
    dp = vec_dot_product( a, b )
    return dp / ( vec_magnitude( a ) * vec_magnitude( b ) )

def saturation_sigmoid( x ):
    """A sample saturation function.

    f(x) = 2.4 / ( 1 + e**(-1.75*x) ) - 1.2
    
    :param x: A value to saturate.
    :type x: float

    :returns: The effect of saturating the input value.
    :rtype: float
    """
    return 2.4/( 1 + e ** (-1.75*x) ) - 1.2
