from __future__ import division, print_function, absolute_import

import sys
from os.path import join
_S = 'src'

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('io',parent_package,top_path)

    config.add_extension('_siesta_io',
                         sources=[join(_S,'write_hsx.f90'),
                                  join(_S,'read_hsx_header.f90'),
                                  join(_S,'read_hsx.f90'),
                                  join(_S,'read_hs_header.f90'),
                                  join(_S,'read_hs.f90'),
                                  join(_S,'read_tshs_header.f90'),
                                  join(_S,'read_tshs.f90'),
                                  join(_S,'read_tshs_header_extra.f90'),
                                  join(_S,'read_tshs_extra.f90'),
                                  join(_S,'read_dm_header.f90'),
                                  join(_S,'read_dm.f90'),
                                  ],
                         )
    #config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
