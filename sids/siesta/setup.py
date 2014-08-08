from __future__ import division, print_function, absolute_import

import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy as np
    config = Configuration('siesta',parent_package,top_path)
    config.add_subpackage('io')
    # Add sparse library
    config.add_extension('sparse',sources=['sparse.c'],
                         include_dirs=['.',np.get_include()])
    return config

if __name__ == '__main__':
    from distutils.core import setup
    setup(**configuration(top_path='').todict())
