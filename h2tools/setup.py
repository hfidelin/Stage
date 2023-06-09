#!/usr/bin/env python
"""
About `h2tools`.
"""

from __future__ import absolute_import, division, print_function

DOCLINES = (__doc__ or '').split("\n")

# Standard library imports
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError
from distutils.util import get_platform
import setuptools
import sys

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log

def generate_a_pyrex_source(self, base, ext_name, source, extension):
    """
    Monkey patch for numpy build_src.build_src method.

    Uses Cython instead of Pyrex, assumes Cython is present.
    """

    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source,
            options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" \
                % (cython_result.num_errors, source))
    return target_file

from numpy.distutils.command import build_src
build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                        assume_default_configuration=True,
                        delegate_options_to_subpackages=True,
                        quiet=True)

    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = ['build/temp%s' % plat_specifier]
    config.add_include_dirs(inc_dir)
    config.get_version('h2tools/__version__.py')

    config.add_subpackage('h2tools')

    return config

def setup_package():
    import setuptools
    from numpy.distutils.core import setup

    metadata = dict(
        name = 'h2tools',
        maintainer = "Alexander Mikhalev",
        maintainer_email = "muxasizhevsk@gmail.com",
        description = DOCLINES[1],
        long_description = DOCLINES[1],
        url = "https://bitbucket.org/muxas/h2tools",
        author = "Alexander Mikhalev",
        author_email = "muxasizhevsk@gmail.com",
        license = 'MIT',
        install_requires = ['numpy>=1.10.1',
            'cython>=0.23.4',
            'maxvolpy>=0.3.6'],
    )

    metadata['configuration'] = configuration
    setup(**metadata)
    return


if __name__ == '__main__':
    setup_package()
