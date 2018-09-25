.PHONY: test
test:
	python3.6 setup.py test

typecheck:
	python3.6 -m mypy --strict --ignore-missing-imports .

typecheck-strict:
	python3.6 -m mypy --strict .

# sdist is a source distribution; bdist_wheel is a "pure Python wheel" (built package)
dist-for-pypi:
	python3.6 generate-README.rst.py
	rm -rf build dist
	python3.6 setup.py sdist
	python3.6 setup.py bdist_wheel

# I forget exactly what the difference between this is and `python3 setup.py test` is
nosetest: typecheck
	nosetests --verbose --nocapture tests
