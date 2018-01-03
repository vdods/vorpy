"""
Use this program to generate README.rst from README.md

You can use http://rst.ninjs.org to view the results.
"""

import sys

# This was suggested by https://stackoverflow.com/questions/26737222/pypi-description-markdown-doesnt-work
try:
    import pypandoc
    pypandoc.convert('README.md', 'rst', extra_args=('--wrap=none', '--'), outputfile='README.rst')
except Exception as e:
    sys.stderr.write('Error while trying to convert README.md to RST format via pypandoc: {0}\n'.format(e))
