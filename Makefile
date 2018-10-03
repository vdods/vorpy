.PHONY: test
test:
	python3.6 setup.py test

.PHONY: typecheck
typecheck:
	python3.6 -m mypy --strict --ignore-missing-imports .

.PHONY: typecheck-strict
typecheck-strict:
	python3.6 -m mypy --strict .

.PHONY: clean
clean:
	rm -rf build dist

README.rst: README.md generate-README.rst.py
	python3.6 generate-README.rst.py

# sdist is a source distribution; bdist_wheel is a "pure Python wheel" (built package)
.PHONY: dist-for-pypi
dist-for-pypi: README.rst
	python3.6 setup.py sdist
	python3.6 setup.py bdist_wheel

# I forget exactly what the difference between this is and `python3 setup.py test` is
.PHONY: nosetest
nosetest: typecheck
	nosetests --verbose --nocapture tests
