import os

PACKAGE_NAME = 'pyflux'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(PACKAGE_NAME, parent_package, top_path)

    config.add_subpackage('__check_build')
    
    config.add_subpackage('arma')
    config.add_subpackage('ensembles')
    config.add_subpackage('families')
    config.add_subpackage('garch')
    config.add_subpackage('gas')
    config.add_subpackage('gpnarx')
    config.add_subpackage('inference')
    config.add_subpackage('output')
    config.add_subpackage('ssm')
    config.add_subpackage('tests')
    config.add_subpackage('var')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())