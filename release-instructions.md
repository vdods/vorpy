# Maintainer Release Instructions

Register accounts with [pypi](https://pypi.org/account/register/) and [testpypi](https://test.pypi.org/account/register/)
(yes, you want both).  From [this](https://packaging.python.org/guides/using-testpypi/#using-test-pypi), note that the
database for TestPyPI may be periodically pruned, so it is not unusual for user accounts to be deleted.

Create the `~/.pypirc` file according to [this documentation](https://packaging.python.org/guides/migrating-to-pypi-org/).
Example:

    [distutils]
    index-servers =
        pypi
        testpypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: jethro-q-walrustitty
    password: raymond-luxury-yacht

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: jethro-q-walrustitty
    password: raymond-luxury-yacht

IMPORTANT: Because this file contains your passwords in plaintext, you will want to set the permissions to as private as possible:

    chmod 600 ~/.pypirc

Omitting the password will cause the package upload to prompt for password.

Create the `setup.py` file and others according to
[this documentation](https://packaging.python.org/tutorials/distributing-packages/#initial-files).
The `long_description` could be the contents of `README.rst` for example.  See [this](setup.py) for an example.

Regarding `README.rst` -- PyPi doesn't support markdown.  If your project has a `README.md` file using markdown (e.g. a
github-hosted project), then you can convert it into `rst` format using [this script](generate-README.rst.py).

Build a "source distribution", as documented [here](https://packaging.python.org/tutorials/distributing-packages/#source-distributions).

    python3 setup.py sdist

This will create a `.tar.gz` file in the `dist` subdir.  Now build a "wheel" (a built version of the package), as documented
[here](https://packaging.python.org/tutorials/distributing-packages/#pure-python-wheels).  For example, to build a pure-Python
wheel for a project that only supports Python 3, run the following command.

    python3 setup.py bdist_wheel

This will create a `.whl` file in the `dist` subdir.

Now upload the distribution files you created to [testpypi](https://testpypi.python.org/pypi/) using `twine`
([why use twine instead of good ol' `setup.py`?](https://pypi.python.org/pypi/twine); also make sure to upgrade
`twine` if necessary):

    python3.6 -m twine upload -r testpypi dist/*

This will make your uploaded files available at

    https://test.pypi.org/simple/<your-package-name>/

Note: [Deleting packages from PyPi](http://comments.gmane.org/gmane.comp.python.distutils.devel/22739).

Now you can test installation of your package via `pip3`.

    pip3 install --index-url https://test.pypi.org/simple/ <your-package-name>

Once you're satisfied with the result, you can upload your package to the "real" index.

    python3.6 -m twine upload dist/*

This will make a project page with relevant info and links available at

    https://pypi.org/project/<your-package-name>/

Finally, your package can be installed as normal:

    pip3 install <your-package-name>

## References

-   https://packaging.python.org/tutorials/distributing-packages/
-   https://packaging.python.org/guides/using-testpypi/#using-test-pypi
