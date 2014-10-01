"""
siesta.io : Enables reading SIESTA fortran files
================================================

Extension to read in fortran binary files outputted from SIESTA
"""

# Enable all fortran codes in this import statement
from ._siesta_io import *
from .ascii import *
from .ascii_tbt import *
