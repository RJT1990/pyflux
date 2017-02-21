import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('arma', parent_package, top_path)

    config.add_extension('arma_recursions',
                         sources=['arma_recursions.c'])
    config.add_extension('nn_architecture',
                         sources=['nn_architecture.c'])

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())