"""
Reading of ordinary SIESTA output files
"""

import numpy as _np
import os as _os, os.path as _osp
import sids.siesta.fdf as _fdf
import sids.helper.units as _unit
import sids.simulation as _sim

class FileASCIIError(Exception):
    """ Error for reading SIESTA files """
    pass

def read(path):
    """
    Read the file name from SIESTA
    
    Parameters
    ==========
    path : the filename (we then guess the file type and read)
    """
    (root,ext) = _osp.splitext(path)
    if ext.lower() == 'fdf':
        return _fdf.FileFDF(path)
    if ext.lower() == 'xv':
        return read_XV(path)
    if ext.lower() == 'fa':
        return read_FA(path)
    if ext.lower() == 'xyz':
        return read_XYZ(path)
    if ext.lower() == 'ani':
        return read_ANI(path)

class XV(_sim.SimulationFile):
    def init_file(self):
        """ Read the XV file
        """
        a = read_XV(self.file_path)
        self.cell = a[0]
        self.vcell = a[1]
        self.xa = a[2]
        self.va = a[3]
        self.Z = a[4]
        self.spec = a[5]
        self.add_clean('cell','vcell','xa','va','Z','spec')

def read_XV(path):
    """
    Reads the XV file and returns the content
    """

    # Conversion factor
    Ang = _unit.UnitConvert('Bohr','Ang')

    with open(path,'r') as fh:
        # Read all lines
        lines = fh.readlines()
        # Read in cell vectors
        cell = _np.empty((3,3),_np.float)
        vcell = _np.empty((3,3),_np.float)
        for i in range(3):
            line = _np.array([float(f) for f in lines.pop(0).split()]) * Ang
            cell[i,:] = line[:3]
            vcell[i,:] = line[3:] # Ang / fs is fine

        # number of atoms
        na = int(lines.pop(0).split()[0])

        # Create data containers
        i = 0
        spec = _np.empty((na,),_np.int)
        Z    = _np.empty((na,),_np.int)
        xa   = _np.empty((na,3),_np.float)
        va   = _np.empty((na,3),_np.float)
        # Read in coordinates
        for line in lines:
            l = line.split()
            spec[i] = int(l.pop(0))
            Z[i] = int(l.pop(0))
            ns = _np.array([float(f) for f in l]) * Ang
            xa[i,:] = ns[:3]
            va[i,:] = ns[3:]
            i += 1
        if i != na:
            raise FileASCIIError('Error in reading file: '+str(path))
    # return a tuple of data
    return cell,vcell,xa,va,Z,spec

class FA(_sim.SimulationFile):
    def init_file(self):
        """ Read the FA file
        """
        self.F = read_FA(self.file_path)
        self.add_clean('F')

def read_FA(path):
    """
    Reads the FA file and returns the content
    """

    # Conversion factor
    Ry = _unit.UnitConvert('Ry','eV')
    Ang = _unit.UnitConvert('Bohr','Ang')
    conv = Ry / Ang

    with open(path,'r') as fh:
        # Read all lines
        lines = fh.readlines()

        na = None
        while na is None:
            line = lines.pop(0).strip()
            if len(line) == 0: continue
            na = int(line)

        fa = _np.empty([na,3],_np.float)
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            l = _np.array([float(f) for f in line.split()])
            fa[int(l[0])-1,:] = l[1:] * conv
        if int(l[0]) != na:
            raise FileASCIIError("Error in FA file")
    return fa

class ANI(_sim.SimulationFile):
    def init_file(self):
        """ Read the ANI file
        """
        self.ani = read_ANI(self.file_path)
        self.add_clean('ani')


def read_ANI(path):
    """
    Reads the ANI file and returns the content
    """

    # Conversion factor
    Ang = _unit.UnitConvert('Bohr','Ang')

    with open(path,'r') as fh:
        # Read all lines
        lines = fh.readlines()
        
        # read number of atoms
        na = None
        while na is None:
            line = lines.pop(0).strip()
            if len(line) == 0: continue
            na = int(line)

        # Placeholder for each animation
        cur = _np.empty([na,3],_np.float)

        i = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            l = _np.array([float(f) for f in line.split()[1:]]) * Ang
            if len(l) == 0: 
                # Check that the number of atoms is correct
                l = int(line)
                if l != na: 
                    raise FileASCIIError("Atoms are changing in the ANI file: "+str(path))
                continue

            # Accumulate data
            cur[i,:] = l
            i += 1
            if i == na:
                # Append new animation step
                cur.shape = (1,na,3)
                try:
                    ani = _np.vstack((ani,cur))
                except:
                    ani = _np.empty([1,na,3],_np.float)
                    ani[0,:,:] = cur
                cur.shape = (na,3)
                i = 0
        if i != 0:
            # the ani file is not complete
            raise FileASCIIError('ANI file: '+str(path)+ ' is not complete')

    return ani

class XYZ(_sim.SimulationFile):
    def init_file(self):
        """ Read the XYZ file
        """
        self.xyz = read_XYZ(self.file_path)
        self.add_clean('xyz')

def read_XYZ(path):
    """
    Reads the XYZ file and returns the content
    """

    # Conversion factor
    Ang = _unit.UnitConvert('Bohr','Ang')

    with open(path,'r') as fh:
        # Read all lines
        lines = fh.readlines()
        
        # read number of atoms
        na = None
        while na is None:
            line = lines.pop(0).strip()
            if len(line) == 0: continue
            na = int(line)
        xyz = _np.empty((na,3),_np.float)

        i = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            l = _np.array([float(f) for f in line.split()[1:]]) * Ang
            xyz[i,:] = l
            i += 1
            if i == na:
                return xyz
        # the ani file is not complete
        raise FileASCIIError('XYZ file: '+str(path)+ ' is not complete')
