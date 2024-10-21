#EXE = ../h3dd_forHsian
EXE = ../h3dd
FF = gfortran
FFLAGS = -O3 -mcmodel=medium

.PHONY: clean
#$(EXE): h3dd_forHsian.o lsqr.o
$(EXE): h3dd.o lsqr.o
	$(FF) $(FFLAGS) -o $@ *.o

%.o: %.f90
	$(FF) $(FFLAGS) -c $<
%.o: %.f
	$(FF) $(FFLAGS) -c $<

clean:
	rm -f *.o *.mod $(EXE)

