import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('gpnarx', parent_package, top_path)

    config.add_extension('kernel_routines',
                         sources=['kernel_routines.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())