"""
Utility for importing DFT directories
=====================================

"""

import os.path as _osp
import glob as _glob

# A class wrapper to be used for DFT-sub-modules
class Simulation(object):
    """
    Object container for DFT simulations

    Imploys a _file and list with containing files.
    As several simulations _can_ be located in one
    single directory we employ a simulation description
    instead
    
    """
    def __init__(self,rules=None,**kwargs):
        """
        Initialize a new directory [DFTDir] which then
        will be traversed by the parent objects init_dir function.
        """
        # We do not force any of the variables in the
        # container
        if 'files' in kwargs: self._files = kwargs['files']
        if 'path' in kwargs: self._path = kwargs['path']
        self._rules = rules # a list of rules
        self._files = {} # type: file
        self.init_simulation()

    def add_file(self,path):
        """ Adds a simulation file to the current simulation """
        for rule in self._rules:
            #print("Trying: "+path+" with rule: "+rule.type)
            if rule.is_file(path):
                #print("Found file: "+rule.type)
                # Create the simulation file
                fS = rule.create_file(path)
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

    def add(self,nattr,attr,overwrite=False):
        """Extends the simulation with an attribute
        """
        if overwrite:
            self.__dict__[nattr] = attr
            return
        if nattr in self.__dict__:
            raise SimulationError("Attribute: "+nattr+" already exists.")

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

class SimulationFile(object):
    """
    A DFT file which has some information attached.
    """
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
    def __init__(self,obj=None,**kwargs):
        """ Initialize a new file rule

        Parameters
 ----------

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
        if 'type' in kwargs:
            self.type = kwargs.pop('type')
        else: self.type = None
        if 'custom' in kwargs:
            self.custom = kwargs.pop('custom')
        # this class is used as an initialization routine
        # when creating the file
        self.obj = obj
        if self.obj is None:
            raise SimulationFileError("Rule has not been passed a class")

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
        return self.obj(path,rule=self,sim=sim)

                


