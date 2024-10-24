EXE = gafocal
SRCDIR = src

.PHONY: run clean
run:
	for d in $(SRCDIR); do (cd $$d; echo $$d; make); done

clean:
	rm -f $(EXE)
	for d in $(SRCDIR); do (cd $$d; echo $$d; make $@); done

