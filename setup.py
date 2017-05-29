import setuptools
import vorpy

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name='vorpy',
    version=vorpy.__version__,
    description='VictOR dods\' PYthon package (mostly math-related)',
    long_description=readme,
    author='Victor Dods',
    author_email='victor.dods@gmail.com',
    url='https://github.com/vdods/vorpy',
    license=license,
    packages=setuptools.find_packages(exclude=('future', 'tests')),
    tests_require=('nose'),
    test_suite='nose.collector'
)
