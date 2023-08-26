# Makefile for creating our standalone Cython program
# Due to https://github.com/cython/cython/tree/master/Demos/embed

PYVERSION=3.5m
PYPREFIX=/usr
INCLUDES=-I$(PYPREFIX)/include/python$(PYVERSION)

test_cython: test_cython.o
	gcc -o $@ $^ -lpython$(PYVERSION)

test_cython.o: test_cython.c
	gcc -c $^ $(INCLUDES)

test_cython.c: test_cython.py
	cython --embed test_cython.py

all: test_cython

clean:
	@echo Cleaning Demos/embed
	@rm -f *~ *.o *.so core core.* *.c test_cython
