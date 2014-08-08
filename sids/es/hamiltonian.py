"""
Electronic structure code which handles the Hamiltonian
"""
import numpy as _np

# Import scipy linear algebra routines
import scipy.linalg as _dlin
import scipy.sparse.linalg as _slin

PI2 = _np.pi * 2.

class Hamiltonian(object):
    """
    Object for retaining information about the Hamiltonian matrix.
    """

    def init_hamiltonian(self,has_overlap):
        """ Initialize the object to the information given"
        
        Parameters
        ==========
        has_overlap : tells the object that an overlap matrix is 
                      accopanying the Hamilton. (i.e. non-orthogonal)
        """
        self.has_overlap = has_overlap

    def eigs(self,k=_np.zeros((3,),_np.float64),method='dense',eigs=None,
             **kwargs):
        """ Return the eigenvalues of this Hamiltonian.
        """
        if method == 'dense':
            if eigs is None: eigs = self.no - 1
            if not isinstance(eigs,tuple):
                eigs = (0,eigs)
            def_opts = {'overwrite_a' : True,
                        'overwrite_b' : True,
                        'eigvals'     : eigs,
                        }
            # We utilize the dense method
            if self.has_overlap:
                H,S = self.todense(k=k)
                def_opts['b'] = S
            else:
                H = self.todense(k=k)
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
            k = max(eigs[0],eigs[1])
            def_opts = {'k' : k,
                        'ncv' : k*3, 
                        'return_eigenvectors' : False,
                        }
            if self.has_overlap:
                H,S = self.tosparse(k=k)
                def_opts['M'] = S
            else:
                H = self.tosparse(k=k)
            def_opts.update(kwargs)
            return _slin.eigvalsh(H,**def_opts)
            
        