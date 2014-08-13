r"""
Utility for importing DFT directories
=====================================

"""

from copy import deepcopy
import os.path as _osp
import glob as _glob
import sids.helper.units as _unit
import numpy as _np

# A class wrapper to be used for DFT-sub-modules
class Simulation(_unit.UnitObject):
    """
    Object container for DFT simulations

    Imploys a _file and list with containing files.
    As several simulations _can_ be located in one
    single directory we employ a simulation description
    instead
    
    """
    _UNITS = _unit.Units()

    def __init__(self,rules=None,**kwargs):
        """
        Initialize a new directory [DFTDir] which then
        will be traversed by the parent objects init_dir function.
        """
        # We do not force any of the variables in the
        # container
        if 'files' in kwargs: 
            self._files = kwargs['files']
        else:
            self._files = {}
        if 'path' in kwargs: 
            self._path = kwargs['path']
        self._rules = rules # a list of rules
        # Start by coping units over (init_simulation
        # can add that in another way
        self._units = deepcopy(self._UNITS)
        self.init_simulation()
    
    # We override the UnitObject convert routine
    def convert(self,unit):
        """ Simulation conversion of units runs through all files
        and converts them """
        # first convert all sitting units
        _unit.UnitObject.convert(self,unit)
        for fo in self._files:
            if isinstance(self._files[fo],list):
                for f in fo:
                    f.convert(unit)
            else:
                self._files[fo].convert(unit)

    def add_file(self,path):
        """ Adds a simulation file to the current simulation """
        for rule in self._rules:
            #print("Trying: "+path+" with rule: "+rule.type)
            if rule.is_file(path):
                #print("Found file: "+rule.type)
                # Create the simulation file (with attached simulation)
                fS = rule.create_file(path,sim=self)

                if rule.vars:
                    # Copy over variables
                    fS.init_file()
                    for v in rule.vars:
                        self.add(v,fS.__dict__[v])
                    fS.clean()
                    
                t = rule.type
                if t in self._files:
                    # We already have a same type
                    # Create a list instead
                    if isinstance(self._files[t],list):
                        self._files[t].append(fS)
                    else:
                        self._files[t] = [self._files[t],fS]
                else:
                    self._files[t] = fS
                pass
            pass

    def add_var(self,name,var,unit=None,overwrite=False,atol=1.e-8):
        """ Add a variable by name, variable and possible unit """
        if overwrite:
            pass
        elif name in self.__dict__:
            if isinstance(var,_np.ndarray):
                if not _np.allclose(self.__dict__[name],var,atol=atol):
                    raise SimulationError("Variable: " + name+" is not the same.")
            else:
                if abs(self.__dict__[name] - var) > atol:
                    raise SimulationError("Variable: " + name+" is not the same.")
            return
        self.__dict__[name] = var
        if unit:
            self._units.append(_unit.Unit(name,unit))

    def add(self,name,attr,overwrite=False):
        """Extends the simulation with an attribute
        """
        if overwrite:
            pass
        elif name in self.__dict__:
            if name != self.__dict__[name]:
                raise SimulationError("Attribute: "+name+" already exists.")
            return
        self.__dict__[name] = attr

    def init_simulation(self):
        """ Default simulation initializer
        Loops over files and appends a list of files
        to the object"""
        if '_path' in self.__dict__:
            for f in _glob.glob(self._path+'/*'):
                self.add_file(f)
        if '_files' in self.__dict__:
            for f in self._files:
                self.add_file(f)

    def get_file(self,type):
        """Returns the type of file that is requested
        """
        if type in self._files:
            # We initialize just before returning (we expect the
            # user will first interact now!)
            if isinstance(self._files[type],list):
                for f in self._files[type]:
                    f.init_file()
            else: 
                self._files[type].init_file()
            return self._files[type]
        return None

class SimulationError(Exception):
    """ Default error handler for simulations """
    pass

class SimulationFile(_unit.UnitObject):
    """
    A DFT file which has some information attached.
    """
    _UNITS = _unit.Units()

    def __init__(self,path,sim=None,**kwargs):
        """ Initialize a file under a DFT directory
        """
        self.sim = sim
        # if a rule exists in kwargs then it must 
        # be that type
        if 'rule' in kwargs:
            self.file_type = kwargs['rule'].type
        elif 'type' in kwargs:
            self.file_type = kwargs['type']
        else:
            raise SimulationFileError("Type not declared **MUST** be declared")
        # Copy over path
        self.file_path = path

        # DO NOT initialize
        # The file will be initialized upon requesting the
        # file from the simulation.

        # If this object has a global entry (_UNITS)
        # we will add that to _units
        self._units = deepcopy(self._UNITS)

        # A container to clean the object from all
        # read in data
        self._cleans = []
        
    def remove_clean(self,*names):
        """ 
        Removes a varible from the clean list
        """
        for name in names:
            for i in range(len(self._cleans)-1,0,-1):
                if self._cleans[i] == name:
                    self._cleans.pop(i)

    def add_clean(self,*names):
        """
        Appends a name of variable to be cleaned if object is cleaned
        """
        for name in names:
            if not name in self._cleans: 
                self._cleans.append(name)
        
    def clean(self):
        """
        Removes all read in data from this object
        This requires the user to have added the data
        to the clean interface
        (we will not delete from the parent simulation)
        """
        for clean in self._cleans:
            try:
                del self.__dict__[clean]
            except: pass

    # If an attribute does not exist on this object
    # check the parent simulation.
    def __getattr__(self, attr):
        if not self.sim is None:
            try:
                return self.sim.__dict__[attr]
            except: pass # fall back to error
        raise SimulationFileError("The attribute: "+attr+" does not exist.")

class SimulationFileError(Exception):
    """ Default error handler for simulation files """
    pass

class RuleFile(object):
    """ 
    Rule for obtaining information about the current simulation.
    """
    def __init__(self,**kwargs):
        """ Initialize a new file rule

        Parameters
        ==========

        obj : the object creator
        ext : search files through the extension of the file
        file : search files through their entire name
        type : save the type of the file (HS,HSX,TSHS)
        custom : dictionary of custom definitions which defines custom rules for this file.
                 Currently we support a "read".
        """
        if 'file' in kwargs:
            self.file = kwargs.pop('file')
            self.is_ext = False
        elif 'ext' in kwargs:
            self.ext = kwargs.pop('ext')
            if self.ext[0] != '.': self.ext = "."+self.ext
            self.is_ext = True
        # Save type of rule-file
        if 'type' in kwargs:
            self.type = kwargs.pop('type')
        else: self.type = None
        if 'custom' in kwargs:
            self.custom = kwargs.pop('custom')
        # this class is used as an initialization routine
        # when creating the file
        self.obj = None
        self.routine = None
        if 'obj' in kwargs:
            self.obj = kwargs.pop('obj')
        elif 'routine' in kwargs:
            self.routine = kwargs.pop('routine')
        if self.obj is None and self.routine is None:
            raise SimulationFileError("Rule has not been passed a class/routine")
        
        # We define a series of variables that 
        # is defined via this variable (i.e. read in data
        # and destroy object immediately)
        if 'variables' in kwargs:
            v = kwargs.pop('variables')
            if isinstance(v,list):
                self.vars = v
            else:
                self.vars = [v]
        else:
            self.vars = []


    def is_file(self,path):
        """
        Checks whether the provided file obeys this rule
        """
        if self.is_ext:
            (root,ext) = _osp.splitext(path)
            # Currently we must enforce that the ending 
            # has correct casing
            if self.ext == ext: return True
        else:
            bname = _osp.basename(path)
            if self.file == bname: return True
        return False

    def create_file(self,path,sim=None):
        """ Creates the file dependent on the rule
        I.e. initializes it from the class contained.
        """
        try:
            return self.obj(path,rule=self,sim=sim)
        except:
            return self.routine(path)


                


