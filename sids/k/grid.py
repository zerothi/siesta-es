
import numpy as _np
import numpy.linalg as _nla

# This helps creates grids for k-point sampling etc.

PI2 = _np.pi * 2.

class KGridException(Exception):
    """ Default error handler for KGrids """
    pass

class Kpoint(_np.ndarray):
    """
    Numpy array with automatic kgrid conversion
    """
    def __new__(cls,k,w):
        """
        Initialize the k-points and weights for this object
        """
        # Create the object we wish to typeset and return
        if k.shape:
            obj = _np.ndarray.__new__(cls,k.shape)
            obj[:] = _np.array(k,dtype=_np.float64)
        else:
            obj = _np.ndarray.__new__(cls,(1,))
            obj[0] = k
        # transfer weights
        obj.w = _np.array(w,dtype=_np.float64)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.w = getattr(obj, 'w', None)

    def __getitem__(self, key):
        """ Overrides the get item so that we 
        also slice the weights
        Possible bug is that w are one dimension smaller than
        k
        """
        nk = super(Kpoint, self).__getitem__(key)
        if isinstance(key,tuple):
            # The user has requested a double slice
            # Only return the k-point value
            #print("getitem " + str(key)+" len "+str(len(self.shape)))
            nkey = (list(key).pop(),)
            w = self.w.__getitem__(nkey)
            return Kpoint(nk,w)
        else:
            if len(self.shape) == 1:
                w = self.w
                return nk
            else:
                w = self.w.__getitem__(key)
        return Kpoint(nk,w)

    def __repr__(self):
        """ Returns a representation of the k-point
        """
        s = ""
        if len(self.shape) > 1:
            # We have a dimensionality
            s = '['
            i = 0
            for k in self:
                s += " " + repr(k) + ','
                i += 1
                if i % 1 == 0:
                    s += '\n '
            if i % 2 == 0: s = s[:-3]

            s += ' ]'
        elif len(self) == 3:
            s = '{{[ {0:7.4f}, {1:7.4f}, {2:7.4f}], {3:7.5f}}}'.format(
                self[0],self[1],self[2],float(self.w))
        else:
            # this will return an array of the quantities
            return repr(_np.array(self))
        return s

    @property
    def nk(self):
        """ Returns number of k-points in this kpoint array
        """
        if len(self.shape) == 1: return 1
        else: 
            return len(self)

    @property
    def k(self):
        """ Returns k-points in this k-point array
        """
        return _np.array(self[:])

    def __getslice__(self, start, stop) :
        """This solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
           Deprecated since version 2.0: Support slice objects as parameters
           to the __getitem__() method. (However, built-in types in CPython
           currently still implement __getslice__(). Therefore, you have to
           override it in derived classes when implementing slicing.)
        """
        return self.__getitem__(slice(start, stop))
    
    def convert(self,rcell):
        """
        Convert all entries in the object to the desired
        units given by the input.
        """
        if len(self) != 3:
            # It makes no sense to convert a k-point
            # without having dimension 3
            raise KGridException("The k-point conversion failed due to bad dimension")
        k = _np.array(self[:])
        return _np.array([
                _np.sum(k[0]*rcell[0,:]),
                _np.sum(k[1]*rcell[1,:]),
                _np.sum(k[2]*rcell[2,:])],dtype=_np.float64)

    def apply_time_reversal(self):
        """ Apply the time-reversal symmetry and remove duplicates
        """
        # make a copy of k to loop them correctly
        rm_list = []
        w = _np.copy(self.w)
        for i in range(self.nk):
            if i in rm_list: continue
            ka = self.k[i]
            kk = _np.delete(self.k,i,axis=0)
            ka = _np.repeat([_np.where(ka == 0.5,ka,-ka)],self.nk-1,axis=0)
            a = _np.all(_np.isclose(kk,ka),axis=1)
            rm = _np.nonzero(a)[0]
            if len(rm) == 0: continue
            rm = rm[0]
            if rm >= i: rm += 1
            if rm not in rm_list:
                w[i] = 2 * w[i]
                rm_list.append(rm)
        # remove unwanted k point
        k = _np.delete(self.k,rm_list,axis=0)
        w = _np.delete(w,rm_list)
        return Kpoint(k,w)


class KPath(object):
    """
    A k-grid path 
    """
    def __init__(self,nk,names,points,loop=True):
        """ Creates a kpath object with respective names and
        points """
        self.nk = nk
        self.names = names
        self.points = _np.array(points)
        if loop:
            self.names.append(names[0])
            self.points = _np.append(self.points,points[0])
            self.points.shape = (-1,3) # correct shape
        self.len = 0.
        for i in range(len(self.points)-1):
            self.len += _np.sqrt(_np.sum((self.points[i+1]-self.points[i])**2))
            
        # We now have the total length
        # create the k-points
        k = _np.empty((0,),_np.float64)
        self.n = [0]
        for i in range(len(self.points)-1):
            l = _np.sqrt(_np.sum((self.points[i+1]-self.points[i])**2))
            # Calculate number of k-points on this path
            n = int(round(l / self.len * nk))
            end = i == len(self.points) - 2
            # update position of label
            self.n.append(self.n[-1]+n)
            k = _np.append(k,k_points(n,self.points[i],self.points[i+1],end=end).flatten())
        self.k = _np.array(k)
        self.k.shape = (-1,3)
        self.nk = len(self.k)
        self.n = _np.array(self.n)

    def get_plot_label(self):
        """ Returns ticks and the labels that should be set
        """
        return self.n,self.names

    def set_plot_label(self,ax):
        """ Returns ticks and the labels that should be set
        """
        t,n = self.get_plot_label()
        ax.set_ticks(t)
        ax.set_ticklabels(n)
        

def k_points(nk,p1,p2,end=False):
    """ Returns k-points which goes from p1 to p2
    with nk points"""
    # Calculate vector going from p1 -> p2
    if nk <= 1:
        raise KGridException("Error in number of points used in k-path (only 0 or 1)")
    np1 = _np.array(p1)
    v = _np.array(p2) - np1
    lv = _np.sqrt(_np.sum(v**2))
    if end:
        k = _np.arange(nk,dtype=_np.float64)/(nk-1)
    else:
        k = _np.arange(nk,dtype=_np.float64)/nk
    return k[:,None] * v[None,:] + np1[None,:]
    

def Monkhorst_Pack(**kwargs):
    """ Creates a Monkhorst-Pack grid using the given options
    
    Parameters
    ==========
    a : number of k-points along a
      optional
    b : number of k-points along b
      optional
    c : number of k-points along c
      optional
    cell : an array containing all information
      optional
    time_reversal : the time reversal symmetry
      optional (default: true)
    displ : the displaced k-grid
    """
    cell = _np.zeros((3,3),_np.int32)
    if 'a' in kwargs: cell[0,0] = kwargs.pop('a')
    if 'b' in kwargs: cell[1,1] = kwargs.pop('b')
    if 'c' in kwargs: cell[2,2] = kwargs.pop('c')
    if 'cell' in kwargs: 
        # get the k-grid cell
        tcell = kwargs.pop('cell')
        if len(tcell.shape) == 2:
            # A full cell has been given
            # (or at least a 3D cell)
            if len(tcell[0]) == 3:
                # a true full cell has been given
                cell = _np.copy(tcell)
            else:
                raise KGridException("The given cell does not conform with the dimensionality, it must be a 3D cell")
        elif len(tcell.shape) == 1:
            # we expect a diagonal cell
            if len(tcell) == 3:
                cell[0,0] = tcell[0]
                cell[1,1] = tcell[1]
                cell[2,2] = tcell[2]
            else:
                raise KGridException("The given cell does not conform with the dimensionality, it must be a 3D diagonal cell")
        else:
            raise KGridException("The given cell does not conform with the dimensionality, it must be a 3D diagonal cell")
    d = _np.zeros((3,),dtype=_np.float64)
    for i in range(3):
        if cell[i,i] % 2 == 0:
            # correct displacement
            d[i] = 1./(cell[i,i]*2)
    if 'displ' in kwargs: d = kwargs.pop('displ')
    d = d - 0.5

    # start calculating the k-grid

    # number of k-points
    nk = int(_np.linalg.det(cell))

    def rep(nk,**kwargs):
        tk = _np.arange(nk,dtype=_np.float64)[::-1]/nk+0.5/nk
        if 'c' not in kwargs:
            tk.shape = (1,1,nk)
            tk = _np.repeat(tk,kwargs.pop('b'),axis=1)
            tk = _np.repeat(tk,kwargs.pop('a'),axis=0)
        elif 'b' not in kwargs:
            tk.shape = (1,nk,1)
            tk = _np.repeat(tk,kwargs.pop('c'),axis=2)
            tk = _np.repeat(tk,kwargs.pop('a'),axis=0)
        elif 'a' not in kwargs:
            tk.shape = (nk,1,1)
            tk = _np.repeat(tk,kwargs.pop('c'),axis=2)
            tk = _np.repeat(tk,kwargs.pop('b'),axis=1)
        return tk.flatten()

    k = _np.empty((nk,3),dtype=_np.float64)
    k[:,0] = rep(cell[0,0],b=cell[1,1],c=cell[2,2]) + d[0]
    k[:,1] = rep(cell[1,1],a=cell[0,0],c=cell[2,2]) + d[1]
    k[:,2] = rep(cell[2,2],b=cell[1,1],a=cell[0,0]) + d[2]
    w = _np.zeros((nk,),dtype=_np.float64) + 1./nk
    
    kpt = Kpoint(k,w)
    if 'time_reversal' in kwargs:
        if kwargs.pop('time_reversal'):
            kpt = kpt.apply_time_reversal()
    elif 'trs' in kwargs:
        if kwargs.pop('trs'):
            kpt = kpt.apply_time_reversal()
    else:
        # We default it to be true
        kpt = kpt.apply_time_reversal()
    return kpt

if __name__ == '__main__':
    kgrid = Kpoint(_np.array([[0.,0.,0.],
                              [0.25,0.,0.]]),
                   _np.array([0.333333,0.666666]))
    k1 = kgrid[0]
    print type(k1),len(k1)
    print repr(k1)
    print k1
    k2 = kgrid[0,0]
    print type(k2),len(k2)
    print repr(k2)
    print k2

    kgrid = Monkhorst_Pack(a=3,b=4,c=2)
    print repr(kgrid)
    kgrid = Monkhorst_Pack(a=33,b=33,c=1)
    print len(kgrid)
    print repr(kgrid)
    kp = KPath(100,
               ['Gamma','X','W','K','U','L'],
               [[0.    , 0.    , 0.    ],
               [1. / 2, 0     , 1. / 2],
               [1. / 2, 1. / 4, 3. / 4],
               [3. / 8, 3. / 8, 3. / 4],
               [5. / 8, 1. / 4, 5. / 8],
               [1. / 2, 1. / 2, 1. / 2]])
    print kp.k,len(kp.k)
