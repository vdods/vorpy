import setuptools
import vorpy

long_description = open('README.md').read()

setuptools.setup(
    name='vorpy',
    version=vorpy.__version__,
    description='VictOR dods\' PYthon package (mostly math-related)',
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Victor Dods',
    author_email='victor.dods@gmail.com',
    url='https://github.com/vdods/vorpy',
    download_url='https://github.com/vdods/vorpy/archive/v{0}.tar.gz'.format(vorpy.__version__),
    keywords=['numerical computation symbolic tensor calculus cached lambdify geometric symplectic integration'],
    license='MIT License',
    packages=setuptools.find_packages(exclude=('future', 'tests')),
    install_requires=list(map(str.strip, open('requirements.txt', 'rt').readlines())),
    python_requires='>=3.6',
    tests_require=('nose'),
    test_suite='nose.collector'
)
