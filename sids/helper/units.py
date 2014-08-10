"""
Library for converting units and creating numpy arrays
with automatic unit conversion.

The conversion factors are taken directly from SIESTA
which means that the number of significant digits are not exact.
"""

import numpy as _np
from copy import copy,deepcopy

_def_L = 'Bohr'
_def_E = 'Ry'
_def_f = 'Ry/Bohr'
_def_T = 'K'
_def_t = 'fs'
_def_M = 'amu'

Bohr = 1.0
Ry   = 1.0
fs   = 1.0
Ang  = 1. / 0.529177
eV   = 1. / 13.60580
hbar = 6.58211928e-16 * eV * 1.e15
kBar = 1. / 1.47108e5
GPa  = kBar * 10.
Kelvin = eV / 11604.45
Debye  = 0.393430
amu    = 2.133107
pi     = 3.14159265358979323846264338327950288419716939937510
deg    = pi / 180.

_ConversionTable = {
    'mass' : {
        'DEFAULT' : _def_M,
        'kg' : 1.,
        'g'  : 1.e-3,
        'amu': 1.66054e-27,
        }, 
    'length' : {
        'DEFAULT' : _def_L,
        'm'    : 1., 
        'cm'   : 0.01, 
        'nm'   : 1.e-9, 
        'Ang'  : 1.e-10, 
        'Bohr' : 0.529177e-10, 
        }, 
    'time' : {
        'DEFAULT' : _def_t,
        's'  : 1. ,
        'fs' : 1.e-15 ,
        'ps' : 1.e-12 ,
        'ns' : 1.e-9 ,
        },
    'energy' : {
        'DEFAULT' : _def_E,
        'J'       : 1., 
        'erg'     : 1.e-7, 
        'eV'      : 1.60219e-19, 
        'meV'     : 1.60219e-22, 
        'Ry'      : 2.17991e-18, 
        'mRy'     : 2.17991e-21, 
        'Hartree' : 4.35982e-18, 
        'K'       : 1.38066e-23, 
        'cm**-1'  : 1.986e-23,
        'kJ/mol'  : 1.6606e-21,
        'Hz'      : 6.6262e-34,
        'THz'     : 6.6262e-22,
        'cm-1'    : 1.986e-23,
        'cm^-1'   : 1.986e-23,
        },
    'force' : {
        'DEFAULT' : _def_f,
        'N'       : 1.,
        'eV/Ang'  : 1.60219e-9,
        'eV/Bohr' : 1.60219e-9*0.529177,
        'Ry/Bohr' : 4.11943e-8,
        'Ry/Ang'  : 4.11943e-8/0.529177,
        }
    }

# from http://physics.nist.gov/PhysRefData/Elements/
__atom = {
     1  : { 'Z' :  1 , 'name' : 'H' , 'amu' : 1.007947 },
     2  : { 'Z' :  2 , 'name' : 'He', 'amu' : 4.002602 },
     3  : { 'Z' :  3 , 'name' : 'Li', 'amu' : 6.9412 },
     4  : { 'Z' :  4 , 'name' : 'Be', 'amu' : 9.012182 },
     5  : { 'Z' :  5 , 'name' : 'B' , 'amu' : 10.8117 },
     6  : { 'Z' :  6 , 'name' : 'C' , 'amu' : 12.01078 },
     7  : { 'Z' :  7 , 'name' : 'N' , 'amu' : 14.00672 },
     8  : { 'Z' :  8 , 'name' : 'O' , 'amu' : 15.99943 },
     9  : { 'Z' :  9 , 'name' : 'F' , 'amu' : 18.9984032 },
    10  : { 'Z' : 10 , 'name' : 'Ne', 'amu' : 20.1797 },
    11  : { 'Z' : 11 , 'name' : 'Na', 'amu' : 22.989770 },
    12  : { 'Z' : 12 , 'name' : 'Mg', 'amu' : 24.30506 },
    13  : { 'Z' : 13 , 'name' : 'Al', 'amu' : 26.9815382 },
    14  : { 'Z' : 14 , 'name' : 'Si', 'amu' : 28.0855 },
    15  : { 'Z' : 15 , 'name' : 'P' , 'amu' : 30.973761 },
    16  : { 'Z' : 16 , 'name' : 'S' , 'amu' : 32.0655 },
    17  : { 'Z' : 17 , 'name' : 'Cl', 'amu' : 35.453 },
    18  : { 'Z' : 18 , 'name' : 'Ar', 'amu' : 39.948 },
    19  : { 'Z' : 19 , 'name' : 'K' , 'amu' : 39.0983 },
    20  : { 'Z' : 20 , 'name' : 'Ca', 'amu' : 40.0784 },
    21  : { 'Z' : 21 , 'name' : 'Sc', 'amu' : 44.955912 },
    22  : { 'Z' : 22 , 'name' : 'Ti', 'amu' : 47.867 },
    23  : { 'Z' : 23 , 'name' : 'V' , 'amu' : 50.9415 },
    24  : { 'Z' : 24 , 'name' : 'Cr', 'amu' : 51.99616 },
    25  : { 'Z' : 25 , 'name' : 'Mn', 'amu' : 54.9380499 },
    26  : { 'Z' : 26 , 'name' : 'Fe', 'amu' : 55.8452 },
    27  : { 'Z' : 27 , 'name' : 'Co', 'amu' : 58.933200 },
    28  : { 'Z' : 28 , 'name' : 'Ni', 'amu' : 58.69342 },
    29  : { 'Z' : 29 , 'name' : 'Cu', 'amu' : 63.5463 },
    30  : { 'Z' : 30 , 'name' : 'Zn', 'amu' : 65.4094 },
    31  : { 'Z' : 31 , 'name' : 'Ga', 'amu' : 69.7231 },
    32  : { 'Z' : 32 , 'name' : 'Ge', 'amu' : 72.64 },
    33  : { 'Z' : 33 , 'name' : 'As', 'amu' : 74.92160 },
    34  : { 'Z' : 34 , 'name' : 'Se', 'amu' : 78.96 },
    35  : { 'Z' : 35 , 'name' : 'Br', 'amu' : 79.904 },
    36  : { 'Z' : 36 , 'name' : 'Kr', 'amu' : 83.798 },
    37  : { 'Z' : 37 , 'name' : 'Rb', 'amu' : 85.4678 },
    38  : { 'Z' : 38 , 'name' : 'Sr', 'amu' : 87.62 },
    39  : { 'Z' : 39 , 'name' : 'Y' , 'amu' : 88.90585 },
    40  : { 'Z' : 40 , 'name' : 'Zr', 'amu' : 91.224 },
    41  : { 'Z' : 41 , 'name' : 'Nb', 'amu' : 92.90638 },
    42  : { 'Z' : 42 , 'name' : 'Mo', 'amu' : 95.96 },
    44  : { 'Z' : 44 , 'name' : 'Ru', 'amu' : 101.07 },
    45  : { 'Z' : 45 , 'name' : 'Rh', 'amu' : 102.90550 },
    46  : { 'Z' : 46 , 'name' : 'Pd', 'amu' : 106.42 },
    47  : { 'Z' : 47 , 'name' : 'Ag', 'amu' : 107.8682 },
    48  : { 'Z' : 48 , 'name' : 'Cd', 'amu' : 112.411 },
    49  : { 'Z' : 49 , 'name' : 'In', 'amu' : 114.818 },
    50  : { 'Z' : 50 , 'name' : 'Sn', 'amu' : 118.710 },
    51  : { 'Z' : 51 , 'name' : 'Sb', 'amu' : 121.760 },
    52  : { 'Z' : 52 , 'name' : 'Te', 'amu' : 127.60 },
    53  : { 'Z' : 53 , 'name' : 'I' , 'amu' : 126.90447 },
    54  : { 'Z' : 54 , 'name' : 'Xe', 'amu' : 131.293 },
    55  : { 'Z' : 55 , 'name' : 'Cs', 'amu' : 132.9054519 },
    56  : { 'Z' : 56 , 'name' : 'Ba', 'amu' : 137.327 },
    57  : { 'Z' : 57 , 'name' : 'La', 'amu' : 138.905477 },
    58  : { 'Z' : 58 , 'name' : 'Ce', 'amu' : 140.116 },
    59  : { 'Z' : 59 , 'name' : 'Pr', 'amu' : 140.90765 },
    60  : { 'Z' : 60 , 'name' : 'Nd', 'amu' : 144.242 },
    62  : { 'Z' : 62 , 'name' : 'Sm', 'amu' : 150.36 },
    63  : { 'Z' : 63 , 'name' : 'Eu', 'amu' : 151.964 },
    64  : { 'Z' : 64 , 'name' : 'Gd', 'amu' : 157.25 },
    65  : { 'Z' : 65 , 'name' : 'Tb', 'amu' : 158.92535 },
    66  : { 'Z' : 66 , 'name' : 'Dy', 'amu' : 162.500 },
    67  : { 'Z' : 67 , 'name' : 'Ho', 'amu' : 164.93032 },
    68  : { 'Z' : 68 , 'name' : 'Er', 'amu' : 167.259 },
    69  : { 'Z' : 69 , 'name' : 'Tm', 'amu' : 168.93421 },
    70  : { 'Z' : 70 , 'name' : 'Yb', 'amu' : 173.054 },
    71  : { 'Z' : 71 , 'name' : 'Lu', 'amu' : 174.9668 },
    72  : { 'Z' : 72 , 'name' : 'Hf', 'amu' : 178.49 },
    73  : { 'Z' : 73 , 'name' : 'Ta', 'amu' : 180.94788 },
    74  : { 'Z' : 74 , 'name' : 'W' , 'amu' : 183.84 },
    75  : { 'Z' : 75 , 'name' : 'Re', 'amu' : 186.207 },
    76  : { 'Z' : 76 , 'name' : 'Os', 'amu' : 190.23 },
    77  : { 'Z' : 77 , 'name' : 'Ir', 'amu' : 192.217 },
    78  : { 'Z' : 78 , 'name' : 'Pt', 'amu' : 195.0782 },
    79  : { 'Z' : 79 , 'name' : 'Au', 'amu' : 196.966552 },
    80  : { 'Z' : 80 , 'name' : 'Hg', 'amu' : 200.59 },
    81  : { 'Z' : 81 , 'name' : 'Tl', 'amu' : 204.3833 },
    82  : { 'Z' : 82 , 'name' : 'Pb', 'amu' : 207.2 },
    83  : { 'Z' : 83 , 'name' : 'Bi', 'amu' : 208.98040 },
    }

# Apply the names to the dictionary so that lookups can be made from index or from names =>
# __atom['He'] == __atom[2]
__atom.update(dict([__atom[k]['name'],v] for k,v in __atom.iteritems()))


#    1001 : 2.016,         # Deuterium
#    2001 : 15.99943,      # FO mix: (1-x) O + x F, x = 0.000
#    2002 : 16.186865825,  # x = 0.063
#    2003 : 16.37430165,   # 0.125
#    2004 : 16.7491733,    # 0.250
#    2005 : 16.59922464,   # 0.200
#    2006 : 16.89912196    # 0.300

# Perhaps this should be alterred into a class so users
# can append other elements?
def AtomMass(atom,unit='amu'): return __atom[atom]['amu'] * UnitConvert('amu',Unit(unit))
def AtomName(atom): return __atom[atom]['name']
def AtomZ(atom): return __atom[atom]['Z']


# Here we start the unit type conversion library
class UnknownUnitTypeError(Exception):
    """
    Error raised when unittype of a unit cannot be found.
    """
    pass # We utilize the generic interface

def UnitType(unit):
    """
    Returns the type of unit that is associated with
    input unit.

    Parameters
    ----------
    unit : str
      unit, e.g. kg, Ang, eV etc. returns the 

    Examples
    --------
    >>> import sids.helper.units as shu
    >>> shu.UnitType('kg')
    'mass'
    >>> shu.UnitType('eV')
    'energy'
    """
    for k in _ConversionTable:
        try:
            if unit['unit'] in _ConversionTable[k]:
                return k
        except:
            try:
                if unit in _ConversionTable[k]:
                    return k
            except:
                pass
    raise UnknownUnitTypeError('The unit "'+str(k)+'" could not be located in the table.')

class UnknownUnitError(Exception):
    """
    Error raised when a unit cannot be found.
    """
    pass # We utilize the generic interface

def UnitConvert(fr,to,opts={}):
    """
    Returns the factor that takes 'fr' to the units of 'to'.

    Parameters
    ----------
    fr :
      starting unit
    to :
      ending unit
    opts :
      controls whether the unit conversion is in powers or fractional units

    Examples
    -------
    >>> import sids.helper.units as shu
    >>> shu.UnitConvert('kg','g')
    1000
    >>> shu.UnitConvert('eV','J')
    1.60219e-19
    """
    # In the case that the conversion to is None, we should do nothing.
    if to is None: return 1.
    fr = Unit(fr) # ensure that it is a unit
    to = Unit(to) # ensure that it is a unit
    frU = None ; toU = None
    frV = None ; toV = None
    
    # Check that the unit types live in the same 
    # space
    # TODO this currently does not handle if powers are taken into
    # consideration.

    for k in _ConversionTable:
        if fr.unit in _ConversionTable[k]:
            frU = k
            frV = _ConversionTable[k][fr.unit]
        if to.unit in _ConversionTable[k]:
            toU = k
            toV = _ConversionTable[k][to.unit]
    if frU != toU:
        raise Exception('The unit conversion is not from the same group: '+frU+' to '+toU)

    # Calculate conversion factor
    val = frV / toV
    for opt in ['^','power','p']:
        if opt in opts: val = val ** opts[opt]
    for opt in ['*','factor','fac']:
        if opt in opts: val = val * opts[opt]
    for opt in ['/','divide','div']:
        if opt in opts: val = val / opts[opt]
    return val


# A single unit-object.
# Contains functions to compare and convert a unit
# to another unit.
class Unit(object):
    """
    Container for the unit and the conversion factors etc.
    This will make it easier to maintain the units, and eventually change the
    usage.
    """
    def __new__(cls,*args,**kwargs):
        if isinstance(args[0],Unit):
            return args[0]
        #print('Creating new unit:',args)
        obj = object.__new__(cls)
        if len(args) == 1: # We are creating a unit without a variable name
            obj.variable = None
            obj.unit = args[0]
        else:
            obj.variable = args[0]
            # Typical case when passing a unit from another variable...
            if isinstance(args[1],Unit):
                obj.unit = args[1].unit
            else:
                obj.unit = args[1]
        
        # We need to handle some type of operator definitions
        # But how to handle them?
        for op in ['**','^','/','*']:
            pass
        
        return obj

    def type(self):
        """ Returns the type of unit this is, i.e. energy, length, time, etc. """
        for k,v in _ConversionTable.iteritems():
            if self.unit in v: return k

    def SI(self):
        """ Returns the SI conversion factor for the unit """
        for k,v in _ConversionTable.iteritems():
            if self.variable in v: return v[self.variable]

    def convert(self,to):
        """ Convert this unit to another and returns the conversion factor. """
        u = Unit(to)
        # This will raise an exception if the units are not of same type...
        conv = UnitConvert(self.unit,u.unit)
        #print('Converting:',self.variable,self.unit,u.unit)
        self.unit = deepcopy(u.unit)
        return conv

    def copy(self):
        """Method for copying the unit """
        return deepcopy(self)

    def __repr__(self):
        """ Return the unit in string format (XML type-like)"""
        return "<Unit variable='"+str(self.variable)+"' unit='"+str(self.unit)+"'/>"

    def __eq__(self,other):
        """ Returns true if the variable is the same as the other """
        return self.variable == other.variable

    def __copy__(self):
        return Unit(copy(self.variable),copy(self.unit))
    
    def __deepcopy__(self, memo):
        return Unit(deepcopy(self.variable),deepcopy(self.unit))


class Units(object):
    """
    Container for many units.
    This will make it easier to maintain the units, and eventually change the
    usage.
    """
    def __new__(cls,*args):
        # Convert the tuple to a list...
        obj = object.__new__(cls)
        # The args are a list of Unit-objects, or a list of pairs which should be converted to a list of units.
        units = []
        i = 0
        while i < len(args):
            if isinstance(args[i],Unit):
                units.append(deepcopy(args[i]))
            else:
                assert i < len(args)-1, 'Can not grap a unit for: ' + str(args[i])
                units.append(deepcopy(Unit(args[i],args[i+1])))
                i += 1
            i += 1
        obj._units = units
        return obj

    def append(self,unit):
        """ Append a unit object """
        # We cannot have to similar units assigned...
        if isinstance(unit,Units):
            for au in unit:
                # Use the recursive routine (keep it simple)
                self.append(au)
        else:
            for u in self:
                if u == unit:
                    raise Exception('Can not append a unit which already exists. Do not assign dublicate variables')
            self._units.append(deepcopy(unit))

    def update(self,unit):
        """ Updates unit object, adds it if it does not exist """
        if unit is None: return
        if isinstance(unit,Units):
            for u in unit:
                self.update(u)
        else:
            for u in self:
                if u.variable == unit.variable:
                    u.unit = deepcopy(unit.unit)
                    return
            self.append(unit)

    def unit(self,variable):
        """ Returns the unit object associated with the variable named variable"""
        # if it is none, return fast.
        if not variable: return None
        for i in self:
            if i.variable == variable:
                return i
        return None

    def copy(self):
        """ Copies this unit segment """
        return deepcopy(self)

    #################
    # General routines overwriting python models
    #################
    def __len__(self):
        return len(self._units)

    def __contains__(self,item):
        if isinstance(item,Unit):
            u = Unit(item.variable,None)
        else:
            u = Unit(item,None)
        for unit in self:
            if u.variable == unit.variable:
                return True
        return False
    
    def __repr__(self):
        """ Return the unit in string format (XML type-like)"""
        tmp = '<Units>'
        for unit in self:
            tmp += '\n  ' + str(unit)
        tmp += '\n</Units>'
        return tmp

    def __iter__(self):
        """ An iterator of the Units collection """
        for unit in self._units:
            yield unit

    def __delitem__(self,variable):
        """ Remove the variable from the units list. """
        for i in range(len(self)):
            if self._units[i].variable == variable:
                del self._units[i]
                return
                
    # We need to overwrite the copy mechanisms.
    # It really is a pain in the ass, but it works.
    # Luckily all copying need only be refered in the Unit-object.
    def __copy__(self):
        units = Units()
        for unit in self:
            units.append(copy(unit))
        return units
    
    def __deepcopy__(self, memo):
        units = Units()
        for unit in self:
            units.append(deepcopy(unit))
        return units

    # Do NOT implement a 'convert' method. It could potentially lead to unexpected behaviour as the
    # Unit-object needs to handle this....
    # TODO consider the conversion of a list of Unit-objects via the Units-object.

class UnitObject(object):
    """
    Contains relevant information about units etc.
    """
    def convert(self,*units):
        """
        Convert all entries in the object to the desired
        units given by the input.
        """
        # Go back in the units variable does not exist.
        if not '_units' in self.__dict__: return

        # If it is a Units object, we can simply loop and do the recursive conversion.
        if isinstance(units[0],Units):
            for unit in units[0]:
                self.convert(unit)
            return

        # First convert all variables associated with a type... ('length',etc.)
        # This well enable one to convert all of length but still have a unit conversion of a
        # single length variable to another.
        for unit in units:
            u = Unit(unit)
            if not u.variable:
                for self_u in self._units:
                    if self_u.type() == u.type():
                        self.__dict__[self_u.variable] *= self_u.convert(u)
                        
        # Now convert the specific requested units.
        for unit in units:
            u = Unit(unit)
            self_u = self.unit(u.variable)
            if self_u:
                self.__dict__[self_u.variable] *= self_u.convert(u)

    def unit(self,variable):
        """ Returns the unit that is associated with the variable """
        return self._units.unit(variable)

    @property
    def units(self):
        """ Returns the units that is associated with the variable """
        return self._units


class Variable_ndarray(_np.ndarray):
    """
    Numpy array with automatic unit conversion.
    
    When two arrays are multiplied we can automatically 
    detect units and convert to the correct units.

    Creating a variable with Variable_ndarray we gain access
    to convert which can convert the unit of the variable.
    """
    def convert(self,unit):
        """
        Convert all entries in the object to the desired
        units given by the input.
        """
        # Go back in the units variable does not exist.
        if not '_units' in self.__dict__: return

        # If it is a Units object, 
        # we can simply loop and do the recursive conversion.
        if isinstance(unit,Units):
            for u in unit: 
                self.convert(u)
            return

        # Ensure that unit is a Unit
        u = Unit(unit)
        
        # Loop over all variables in this object.
        # It only has one
        for i in self._units:
            if i.type() == u.type():
                self[:] *= i.convert(u)

    def add_unit(self,var,unit):
        """ Adds a unit to a variable beloning to the object """
        

    def unit(self,variable='self'):
        """ Returns the unit that is associated with the variable """
        return self._units.unit(variable)

    @property
    def units(self):
        """ Returns the units that is associated with the variable """
        return self._units

    @staticmethod
    def _N(array):
        return _np.array(array)

    def __array_finalize__(self,obj):
        """ Finalize the array with the object """
        if obj is None: return

        # Create the default units, we need to copy them, to ensure
        # that we do not attach the same objects.
        if hasattr(obj,'_units'):
            self._units = deepcopy(obj._units)
        else:
            self._units = deepcopy(self._UNITS)

        if hasattr(self,'__variable_finalize__'):
            self.__variable_finalize__()
