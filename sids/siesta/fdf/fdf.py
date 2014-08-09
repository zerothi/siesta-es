"""
Implements the fdf key word algorithm
"""

import sids.helper.units as _unit
from copy import deepcopy

# The boolean values that FDF system utilizes
_BOOL = { 't' : True, 'true' : True,
          '.true.' : True, 'yes' : True,
          'y' : True,
          'f' : False, 'false' : False,
          '.false.' : False, 'no' : False,
          'n' : False
          }

_DEL = [':','>','=','[',']']

class FDFException(Exception):
    """
    Error raised when an error occured in an FDF key
    """
    pass # We utilize the generic interface

class FileFDF(object):
    """ An object for retaining the fdf keywords
    """
    
    def __init__(self,file_path):
        """ input file for reading """
        self.file_path = file_path

        # Populate the dictionary
        self._fdf = FileFDF.read(file_path)

    def __len__(self):
        """ returns number of keys in fdf file """
        return len(self._fdf)

    @staticmethod
    def get_line(line):
        """ Parse the line """
        i = line.find('#')
        if i < 0: i = line.find('!')
        if i < 0: i = line.find(';')
        if i == 0: return []
        if i > 0:
            line = line[:i]
            if len(line.strip()) == 0: return []
        # Now we are ready to figure out what is going on
                
        # split to segments (including all other known
        # separators)
        ls = line.split()
        for sep in _DEL:
            l = []
            for lls in ls:
                l.extend(lls.split(sep))
            ls = l

        for sep in ['"',"'"]:
            l = []
            # concatenate strings

        # remove all empty statements
        ls = filter(None,ls)
        return ls

    def __getattr__(self,attr):
        """ Returns the value of the FDF key """
        if attr.lower() in self._fdf: return self._fdf[attr.lower()]
        raise FDFException("Key does not exist: "+str(attr))

    @staticmethod
    def read(file_path):
        """ Reads in the fdf file and returns a dictionary
        containing all keys in the fdf file """

        def read_pipe(file_name):
            block = []
            with open(file_name,'r') as ffh:
                for line in ffh.readlines():
                    ls = FileFDF.get_line(line)
                    if len(ls) == 0: continue
                    block.append(ls)
            return block
        
        d = {}
        with open(file_path,'r') as fh:
            in_block = None
            for line in fh.readlines():
                ls = FileFDF.get_line(line)
                
                # quick skip for zero length
                if len(ls) == 0: continue

                # Now we have all segments
                # check for include statements
                if ls[0].lower() == '%include':
                    # Read next file
                    d1 = FileFDF.read(ls[1])
                    d1.update(d)
                    d = deepcopy(d1)
                    continue

                if in_block:
                    if ls[0].lower() == '%endblock':
                        if in_block not in d:
                            d[in_block] = block
                        in_block = None
                        continue
                    block.append(ls)
                    continue

                # lower case the first element
                ls[0] = ls[0].lower()

                if ls[0] == '%block':
                    # get block-name
                    in_block = ls[1].lower()
                    if len(ls) > 2:
                        if ls[2] == '<':
                            # We have a piped in file
                            ls[3] = ls[3].strip('"').strip("'")
                            block = read_pipe(ls[3])
                    
                            if in_block not in d: 
                                d[in_block] = block
                                in_block = None
                            continue

                    block = []
                    continue
                elif len(ls) > 1:
                    if ls[1] == '<':
                        # We have a piped in file
                        ls[2] = ls[2].strip('"').strip("'")
                        d1 = FileFDF.read(ls[2])
                        d1.update(d)
                        d = d1
                        continue
                
                # We are sure that we now have
                # a simple keyword
                if ls[0] in d: continue

                # set keyword
                d[ls[0]] = ls[1:]

        return d

    def get(self,key,parse='u'):
        """ Returns keyword """
        return fdf.line_parse(fdf.__getattr__(key),parse)

    @staticmethod
    def line_parse(line,parse):
        """ Parses the line and returns the output 

        Parameters
        ==========
        line : input list of tokens in the line
        parse : string of lines
        """
        if not isinstance(parse,str):
            # We have a parsing object
            pass

        lp = []
        i = 0
        for p in parse:
            if p == 'n':
                lp.append(line[i])
            elif p == 'r':
                lp.append(float(line[i]))
            elif p == 'i':
                lp.append(int(line[i]))
            elif p == 'u':
                t = _unit.UnitType(line[i+1])
                # Get default
                d = _unit._ConversionTable[t]['DEFAULT']
                lp.append(float(line[i])*_unit.UnitConvert(line[i+1],d))
                i += 1
            else:
                raise FDFException("Parse error for FDF line")
            i += 1
        if len(lp) == 1: return lp[0]
        return lp

if __name__ == "__main__":
    fdf = FileFDF("sample.fdf")
    print fdf.line_parse(fdf.MeshCutoff,'u')
    print fdf.get('MeshCutoff','u')
    print fdf._fdf
