import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('inference', parent_package, top_path)

    config.add_extension('metropolis_sampler',
                         sources=['metropolis_sampler.c'])
    config.add_extension('bbvi_routines',
                         sources=['bbvi_routines.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())