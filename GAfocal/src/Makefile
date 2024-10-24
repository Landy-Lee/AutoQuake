EXE = ../gafocal
FF = gfortran
FFLAGS = -O3 -mcmodel=medium

.PHONY: clean
$(EXE): GaFpfit.o
	$(FF) $(FFLAGS) -o $@ *.o

%.o: %.f90
	$(FF) $(FFLAGS) -c $<

clean:
	rm -f *.o $(EXE)

