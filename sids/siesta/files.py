"""
A module that defines all SIESTA files known
"""

import sids.simulation as _sim
import sids.siesta.io as _sio
import sids.es as _es
import numpy as _np
import scipy.sparse as _spar
import sparse as spar

class SiestaHamilton(_sim.SimulationFile,_es.Hamiltonian):
    """ A wrapper class to ease the construction of several
    Hamiltonian formats.
    """
    def todense(self,k=_np.zeros((3,),_np.float64),E=None,spin=0,name=None):
        """ Returns a dense matrix of this Hamiltonian at the specified
        k-point"""
        
        # convert k-point to current cell size
        tk = _es.PI2 * _np.dot(k,self.rcell)
        if name is None:
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.H[spin,:],self.S)
        elif name == 'H':
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.H[spin,:])
        elif name == 'S':
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.S)
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
        self.list_ptr = _np.require(list_ptr,requirements=['C','A'])
        del list_ptr
        self.list_col = _np.require(list_col,requirements=['C','A']) - 1 # correct numpy indices
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

    def _correct_sparsity(self):
        """ Corrects the xij array and utilizes offsets
        instead."""
        # Correct the xij array (remove xa[j]-xa[i])
        spar.xij_correct(self.na, self.xa, self.lasto,
                         self.no, self.n_col, self.list_ptr, self.list_col,
                         self.xij)

        # get transfer matrix sizes
        tm = spar.xij_sc(self.rcell,self.nnzs,self.xij)

        # The supercell offsets (in Ang)
        self.offset = spar.get_supercells(self.cell, tm)

        # Get the integer offsets for all supercells
        ioffset = spar.get_isupercells(tm)

        # Correct list_col (create the correct supercell index)
        spar.list_col_correct(self.rcell, self.no, self.nnzs, 
                              self.list_col, self.xij, 
                              tm, ioffset)
    
class HSX(SiestaHamilton):
    """ The HSX file that contains the Hamiltonian, overlap and
    xij
    """
    def init_file(self):
        """ Initialization of the HSX file data type
        """
        self.init_HSxij(_sio.read_hsx_header,_sio.read_hsx)
    

class HS(SiestaHamilton):
    """ The HS file that contains the Hamiltonian, overlap and
    xij
    """
    def init_file(self):
        """ Initialization of the HS file data type
        """
        self.init_HSxij(_sio.read_hs_header,_sio.read_hs)

class TSHS(SiestaHamilton):
    """ The TSHS file that contains the Hamiltonian, overlap and
    xij
    """
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
            # We add the cell size to the simulation
            self.sim.add('cell',self.cell)
        except: pass
        self.lasto, xa = \
            _sio.read_tshs_extra(self.file_path,na_u=self.na)
        # Convert xa to C-array
        self.xa = _np.require(xa.T,requirements=['C','A'])
        self.xa.shape = (self.na,3)
        del xa

        # create offsets 
        self._correct_sparsity()
        del self.xij            


class SiestaDensityMatrix(_sim.SimulationFile):
    """ A wrapper class to ease the construction of several
    Hamiltonian formats.
    """
    def todense(self,k=_np.zeros((3,),_np.float64),spin=0,name=None):
        """ Returns a dense matrix of this Hamiltonian at the specified
        k-point"""
        
        # convert k-point to current cell size
        tk = _es.PI2 * _np.dot(k,self.rcell)
        if name is None:
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.DM[spin,:],self.EM)
        elif name == 'DM' or name == 'D':
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.DM[spin,:])
        elif name == 'EM' or name == 'E':
            return spar.todense_off(tk,self.no,
                                 self.n_col,self.list_ptr,self.list_col,
                                 self.offset,self.EM)
        else:
            raise Exception("Error in name")


class DM(SiestaDensityMatrix):
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
        self.list_ptr = _np.require(list_ptr,requirements=['C','A'])
        del list_ptr
        self.list_col = _np.require(list_col,requirements=['C','A']) - 1 # correct numpy indices
        del list_col
        self.DM = _np.require(DM.T,requirements=['C','A'])
        del DM
        self.DM.shape = (self.nspin,self.nnzs)
        # Done reading in information