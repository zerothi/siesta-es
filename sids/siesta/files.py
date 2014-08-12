"""
A module that defines all SIESTA files known
"""
from copy import deepcopy

import sids.simulation as _sim
import sids.siesta.io as _sio
import sids.es as _es
import sids.helper.units as _unit
import sids.k as _k
import numpy as _np
import scipy.sparse as _spar
import sparse as spar

class SparseMatrixError(Exception):
    """
    Error handler for SIESTA sparse matrices
    """
    pass

class SparseMatrix(_sim.SimulationFile):
    """ 
    A wrapper class for the sparsity matrices in siesta
    """

    def _option(self,method='dense'):
        """
        Sets specific options that determines the working of this
        sparse matrix.
        
        Parameters
        ----------
        method -- 'dense' or 'sparse' enables choice of algorithms used
        """
        self.method = method
        if method == 'dense':
            try:
                del self.s_ptr
                del self.s_col
            except:
                pass
        elif method == 'sparse':
            self.s_ptr, self.s_col = spar.sparse_uc(self.no, self.n_col, self.l_ptr,
                                                    self.l_col)

    def option(self,**opt):
        """
        Enables specification options set.
        As this should be extendable we set this as a method to be overwritten by 
        users.
        Just remember to end your routine with "self._set(**opts)"
        """
        self._option(**opt)

    def _tosparse(self,k,m1,m2=None):
        """
        Returns a csr sparse matrix at the specified k-point
        """

        if not hasattr(self,'s_ptr'):
            raise SparseMatrixError("Sparse method have not been initialized, call self.set(method='sparse')")

        # convert k-point to current cell size
        tk = _k.PI2 * _np.dot(k,self.rcell)
        if hasattr(self,'offset'):
            return spar.tosparse_off(tk,self.no,
                                     self.n_col,self.l_ptr,self.l_col,
                                     self.offset,self.s_ptr,self.s_col,m1,m2)
        else:
            return spar.tosparse(tk,self.no,
                                 self.n_col,self.l_ptr,self.l_col,
                                 self.xij,self.s_ptr,self.s_col,m1,m2)

    def _todense(self,k,m1,m2=None):
        """ Returns a dense matrix of this Hamiltonian at the specified
        k-point"""
        
        # convert k-point to current cell size
        tk = _k.PI2 * _np.dot(k,self.rcell)
        if hasattr(self,'offset'):
            return spar.todense_off(tk,self.no,
                                    self.n_col,self.l_ptr,self.l_col,
                                    self.offset,m1,m2)
        else:
            return spar.todense(tk,self.no,
                                self.n_col,self.l_ptr,self.l_col,
                                self.xij,m1,m2)

    def _correct_sparsity(self):
        """ 
        Corrects the xij array and utilizes offsets instead.
        """
        # Correct the xij array (remove xa[j]-xa[i])
        spar.xij_correct(self.na, self.xa, self.lasto,
                         self.no, self.n_col, self.l_ptr, self.l_col,
                         self.xij)

        # get transfer matrix sizes
        tm = spar.xij_sc(self.rcell,self.nnzs,self.xij)

        # The supercell offsets (in Ang)
        self.offset = spar.get_supercells(self.cell, tm)

        # Get the integer offsets for all supercells
        ioffset = spar.get_isupercells(tm)

        # Correct list_col (create the correct supercell index)
        spar.list_col_correct(self.rcell, self.no, self.nnzs, 
                              self.l_col, self.xij, 
                              tm, ioffset)

class Hamiltonian(SparseMatrix,_es.Hamiltonian):
    """ A wrapper class to ease the construction of several
    Hamiltonian formats.
    """
    # Default units (we convert in the FORTRAN routines)
    _UNITS = _unit.Units('H','eV','xij','Ang')

    def tosparse(self,k=_np.zeros((3,),_np.float64),spin=0,name=None):
        """ Returns a sparse matrix of this Hamiltonian at the specified
        k-point"""
        if name is None:
            return self._tosparse(k,self.H[spin,:],self.S)
        elif name == 'H':
            return self._tosparse(k,self.H[spin,:])
        elif name == 'S':
            return self._tosparse(k,self.S)
        else:
            raise SparseMatrixError("Error in name")

    def todense(self,k=_np.zeros((3,),_np.float64),spin=0,name=None):
        """ Returns a dense matrix of this Hamiltonian at the specified
        k-point"""
        if name is None:
            return self._todense(k,self.H[spin,:],self.S)
        elif name == 'H':
            return self._todense(k,self.H[spin,:])
        elif name == 'S':
            return self._todense(k,self.S)
        else:
            raise Exception("Error in name")

    def init_HSxij(self,read_header,read):
        """ Initialization of the HSX file data type
        """

        # Initialize the Hamiltonian object
        self.init_hamiltonian(has_overlap=True)

        self.gamma,self.nspin, self.no, self.no_s, \
            self.nnzs = read_header(self.file_path)
        self.gamma = self.gamma != 0
        n_col,list_ptr,list_col,H,S,xij = \
            read(fname=self.file_path,
                 gamma=self.gamma,
                 no_u=self.no,no_s=self.no_s,
                 maxnh=self.nnzs,nspin=self.nspin)
        # Correct contiguous
        self.n_col = _np.require(n_col,requirements=['C','A'])
        del n_col
        self.l_ptr = _np.require(list_ptr,requirements=['C','A'])
        del list_ptr
        self.l_col = _np.require(list_col,requirements=['C','A']) - 1 # correct numpy indices
        del list_col
        self.H = _np.require(H.T,requirements=['C','A'])
        del H
        self.H.shape = (self.nspin,self.nnzs)
        self.S = _np.require(S,requirements=['C','A'])
        del S
        self.xij = _np.require(xij.T,requirements=['C','A'])
        del xij
        self.xij.shape = (self.nnzs,3)
        # Done reading in information
    
class HSX(Hamiltonian):
    """ The HSX file that contains the Hamiltonian, overlap and
    xij
    """
    def init_file(self):
        """ Initialization of the HSX file data type
        """
        self.init_HSxij(_sio.read_hsx_header,_sio.read_hsx)

class HS(Hamiltonian):
    """ The HS file that contains the Hamiltonian, overlap and
    xij
    """
    def init_file(self):
        """ Initialization of the HS file data type
        """
        self.init_HSxij(_sio.read_hs_header,_sio.read_hs)

class TSHS(Hamiltonian):
    """ The TSHS file that contains the Hamiltonian, overlap and
    xij
    """
    _UNITS = _unit.Units('H','eV','cell','Ang',
                   'xa','Ang','Ef','eV')
    def init_file(self):
        """ Initialization of the TSHS file data type
        """
        self.init_HSxij(_sio.read_tshs_header,_sio.read_tshs)
        # Read extra information contained in TSHS
        self.na, cell, self.Ef, self.Qtot, self.T = \
            _sio.read_tshs_header_extra(self.file_path)
        self.cell = _np.require(cell.T,requirements=['C','A'])
        self.cell.shape = (3,3)
        del cell
        self.rcell = _np.linalg.inv(self.cell)
        try:
            self.sim.add('na',self.na)
        except: pass
        try:
            # We add the cell size to the simulation
            self.sim.add_var('cell',self.cell,self._units.unit('cell'))
            self.sim.add('rcell',self.rcell)
        except: pass
        self.lasto, xa = \
            _sio.read_tshs_extra(self.file_path,na_u=self.na)
        try:
            self.sim.add('lasto',self.lasto)
        except: pass
        
        # Convert xa to C-array
        self.xa = _np.require(xa.T,requirements=['C','A'])
        self.xa.shape = (self.na,3)
        del xa
        try:
            self.sim.add_var('xa',self.xa,self._units.unit('xa'))
        except: pass

        # create offsets 
        self._correct_sparsity()
        del self.xij            

class DensityMatrix(SparseMatrix):
    """ A wrapper class to ease the construction of several
    Hamiltonian formats.
    """
    def todense(self,k=_np.zeros((3,),_np.float64),spin=0,name='D'):
        """ Returns a dense matrix of this Hamiltonian at the specified
        k-point"""
        if name is None:
            return self._todense(k,self.DM[spin,:],self.EM)
        elif name in ['DM','D']:
            return self._todense(k,self.DM[spin,:])
        elif name in ['EM','E']:
            return self._todense(k,self.EM)
        else:
            raise Exception("Error in name")

class DM(DensityMatrix):
    """The density matrix file
    """
    def init_file(self):
        """ Initialization of the DM file data type
        """
        self.nspin, self.no, self.nnzs = _sio.read_dm_header(self.file_path)
        n_col,list_ptr,list_col,DM = \
            _sio.read_dm(fname=self.file_path,
                 no_u=self.no, maxnd=self.nnzs,nspin=self.nspin)
        # Correct contiguous
        self.n_col = _np.require(n_col,requirements=['C','A'])
        del n_col
        self.l_ptr = _np.require(list_ptr,requirements=['C','A'])
        del list_ptr
        self.l_col = _np.require(list_col,requirements=['C','A']) - 1 # correct numpy indices
        del list_col
        self.DM = _np.require(DM.T,requirements=['C','A'])
        del DM
        self.DM.shape = (self.nspin,self.nnzs)
        # Done reading in information
