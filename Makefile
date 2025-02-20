# Makefile for Pyclassify

.PHONY: make
make:
	python src/pyclassify/_compile.py
	python -m pip install -e .

# clean up
.PHONY: clean
clean:
	python -m pip uninstall pyclassify -y
