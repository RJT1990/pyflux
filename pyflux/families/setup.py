import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('families', parent_package, top_path)

    config.add_extension('gas_recursions',
                         sources=['gas_recursions.c'])
    config.add_extension('poisson_kalman_recursions',
                         sources=['poisson_kalman_recursions.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())