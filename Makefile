# sdist is a source distribution; bdist_wheel is a "pure Python wheel" (built package)
dist-for-pypi:
	python3 generate-README.rst.py
	rm -rf build dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test:
	python3 setup.py test

# I forget exactly what the difference between this is and `python3 setup.py test` is
nosetest:
	nosetests --verbose --nocapture tests
