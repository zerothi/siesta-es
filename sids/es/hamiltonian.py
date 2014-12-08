"""
Electronic structure code which handles the Hamiltonian
"""
import numpy as _np
import sids.helper.units as _unit

# Import scipy linear algebra routines
import scipy.linalg as _dlin
import scipy.sparse.linalg as _slin

class Hamiltonian(object):
    """
    Object for retaining information about the Hamiltonian matrix.
    """

    def init_hamiltonian(self,ortho):
        """ Initialize the object to the information given"
        
        Parameters
        ==========
        ortho : tells the object whether it has an overlap matrix
                      accompanying the Hamilton. (i.e. non-orthogonal)
        """
        self.ortho = ortho
        self.method = 'dense'

    def eigs(self,k=_np.zeros((3,),_np.float64),eigs=None,**kwargs):
        """ Return the eigenvalues of this Hamiltonian.
        """
        try:
            method = self.method
        except:
            method = 'dense'
        if method == 'dense':
            if eigs is None: eigs = self.no - 1
            if not isinstance(eigs,tuple):
                eigs = (0,eigs)
            def_opts = {'overwrite_a' : True,
                        'overwrite_b' : True,
                        'eigvals'     : eigs,
                        }
            # We utilize the dense method
            if self.ortho:
                H = self.todense(k=k)
            else:
                H,S = self.todense(k=k)
                def_opts['b'] = S
            # Update user-given options
            def_opts.update(kwargs)

            return _dlin.eigvalsh(H,**def_opts)

        elif method == 'sparse':
            if eigs is None: eigs = 2*self.no/3
            # Using the ARNOLDI method we cannot
            # compute all eigenvalues.
            # Hence we must truncate at some number
            # If the user is using the default
            # eigvalsh will error out
            # The user must ensure a limited number of 
            # eigenvalues to be calculated
            if not isinstance(eigs,tuple):
                neig = eigs
            else:
                neig = max(eigs[0],eigs[1])
            def_opts = {'k' : neig,
                        'ncv' : neig*3, 
                        'return_eigenvectors' : False,
                        }
            if self.ortho:
                H = self.tosparse(k=k)
            else:
                H,S = self.tosparse(k=k)
                def_opts['M'] = S
            def_opts.update(kwargs)
            return _slin.eigsh(H,**def_opts)
            
        
    def bandstructure(self,kpath):
        """ Calculates the bandstructure of the Hamiltonian
        by passing a kpath. """

        es = _np.zeros((kpath.nk,self.no),_np.float)

        i = 0
        for k in kpath.k:
            # Calculate eigenvalues
            e = self.eigs(k=k)
            es[i,:len(e)] = e
            i += 1
        return es

    def plot_bandstructure(self,kpath,Emin=None,Emax=None,axis=None):
        """ Plots the bandstructure of this Hamiltonian
        You can specify the ranges you wish to populate.
        """
        es = self.bandstructure(kpath)
        import matplotlib.pyplot as plt
        if axis is None:
            f,axis = plt.subplots()
        axis.plot(_np.arange(kpath.nk),es)
        kpath.set_plot_label(axis.xaxis)
        axis.set_ylim([Emin,Emax])
        return es,axis

