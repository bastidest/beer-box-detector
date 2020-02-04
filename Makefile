VPATH := ./ELSDc/src

libELSDc: pgm.c svg.c elsdc.c gauss.c curve_grow.c polygon.c ring.c ellipse_fit.c rectangle.c iterator.c image.c lapack_wrapper.c misc.c
	cc -O3 -o $@ $^ -llapack -lm -shared -fPIC

clean:
	rm libELSDc
