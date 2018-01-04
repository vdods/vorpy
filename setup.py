import setuptools
import vorpy

long_description = open('README.rst').read()
license = open('LICENSE').read()

setuptools.setup(
    name='vorpy',
    version=vorpy.__version__,
    description='VictOR dods\' PYthon package (mostly math-related)',
    long_description=long_description,
    author='Victor Dods',
    author_email='victor.dods@gmail.com',
    url='https://github.com/vdods/vorpy',
    download_url='https://github.com/vdods/vorpy/archive/v{0}.tar.gz'.format(vorpy.__version__),
    keywords=['numerical computation symbolic tensor calculus cached lambdify geometric symplectic integration'],
    license=license,
    packages=setuptools.find_packages(exclude=('future', 'tests')),
    install_requires=[
        'dill',
        'numpy',
        'sympy'
    ],
    python_requires='>=3',
    tests_require=('nose'),
    test_suite='nose.collector'
)
