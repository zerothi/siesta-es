from __future__ import division, print_function, absolute_import

import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy as np
    config = Configuration('es',parent_package,top_path)
    # Add self-energy calculation ( add f2py header in top dir )
    config.add_extension('self_energy',sources=['self_energy.c'],
                         include_dirs=['.','..',np.get_include()])
    #config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
