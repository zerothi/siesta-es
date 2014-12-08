"""
Electronic structure code which uses a Hamiltonian to handle Green's functions
"""
import numpy as _np
import sids.helper.units as _unit

import hamiltonian as ham
import self_energy as se

class SelfEnergy(ham.Hamiltonian):
    """
    Object for retaining information regarding the
    self-energies from Hamiltonians.
    """
    t_dir = 2

    def init_SE(self,dir=2):
        """
        Set the self-energy
        """
        self.t_dir = dir

    def set_TM(self,k=_np.zeros((3,),_np.float),spin=0):
        """
        Sets the transfer matrices for this k-point
        """
        kt = _np.copy(k)
        if self.ortho:
            H = self.H(k=kt,spin=spin)
            kt[self.t_dir] = 0.5
            H5 = self.H(k=kt,spin=spin)
            self.H0 = .5 * (H + H5)
            del H5
            kt[self.t_dir] = 0.25
            H25 = self.H(k=kt,spin=spin)
            self.H1 = .5 * ( H - self.H0 - 1j * (H25 - self.H0) )
            del H, H25
            self.H0 = _np.require(self.H0,requirements=['C','A'])
            self.H1 = _np.require(self.H1,requirements=['C','A'])
        else:
            HS = self.H(k=kt,spin=spin)
            kt[self.t_dir] = 0.5
            HS5 = self.H(k=kt,spin=spin)
            self.H0 = 0.5 * (HS[0] + HS5[0])
            self.S0 = 0.5 * (HS[1] + HS5[1])
            del HS5
            kt[self.t_dir] = 0.25
            HS25 = self.H(k=kt,spin=spin)
            self.H1 = .5 * ( HS[0] - self.H0 - 1j * (HS25[0] - self.H0) )
            self.S1 = .5 * ( HS[1] - self.S0 - 1j * (HS25[1] - self.S0) )
            del HS, HS25
            self.H0 = _np.require(self.H0,requirements=['C','A'])
            self.H1 = _np.require(self.H1,requirements=['C','A'])
            self.S0 = _np.require(self.S0,requirements=['C','A'])
            self.S1 = _np.require(self.S1,requirements=['C','A'])

    def clean_TM(self):
        """
        Cleans internal data structures for transfer matrices
        """
        if not self.ortho: del self.S0, self.S1
        del self.H0, self.H1

    def SE(self,E,method='lopez',eps=1.e-15):
        """
        Returns the self-energy of this Hamiltonian
        
        Parameters
        ==========
        E: the energy of this self-energy calculation
        k: the k-point for this self-energy calculation
        method: the method used to solve the self-energy.
        dir: the direction in the unit-cell direction 
             which is semi-infinite
        """
        # We need the energy with respect to the 
        # Fermi level
        tE = E + self.Ef
        # First create the transfer matrices
        if self.ortho:
            return se.self_energy_ortho(tE,self.H0,self.H1,
                                        method=method,eps=eps)
        return se.self_energy(tE,self.H0,self.S0,self.H1,self.S1,
                              method=method,eps=eps)

    
