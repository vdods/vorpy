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
    download_url='https://github.com/vdods/vorpy/archive/v{0}.tar.gz'.format(vorpy.__version__),
    keywords=['numerical computation symbolic tensor calculus cached lambdify geometric symplectic integration'],
    license=license,
    packages=setuptools.find_packages(exclude=('future', 'tests')),
    tests_require=('nose'),
    test_suite='nose.collector'
)
