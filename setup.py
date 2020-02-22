#!/usr/bin/env python
"""
Analysis of continuous time Markov chains on trees.

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup


setup(
        name='jsonctmctree',
        version='0.2.0',
        description=DOCLINES[0],
        author='jsonctmctree authors',
        url='https://github.com/xji3/jsonctmctree/',
        download_url='https://github.com/xji3/jsonctmctree/',
        packages=['jsonctmctree', 'jsonctmctree.pyexp', 'jsonctmctree.tests'],
        #test_suite='nose.collector',
        #package_data={'jsonctmctree' : ['tests/test_*.py']},
        scripts=[
            'scripts/jsonctmctree',
            'scripts/jsonctmctree-ll',
            'scripts/jsonctmctree-expect',
            ]
        )
