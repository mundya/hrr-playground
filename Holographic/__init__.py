"""A Library for Experimenting with Holographic Reduced Representations

.. moduleauthor:: Andrew Mundy <mundya@cs.manchester.ac.uk>

Provides a useful set of tools for experimenting with ideas using HRRS,
particularly where you don't want to go to the lengths of building a 
nengo example.
"""

import Memory
import Symbol
from utils import ( vec_generate, vec_convolve_circular, vec_exponentiate,
                    vec_magnitude )
