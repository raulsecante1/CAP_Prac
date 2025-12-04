EXECS=heat
MPICC=mpicc

all:
	${MPICC} -o ${EXECS} heat.c -lm -fopenmp

run:
	mpirun -np 4 ./${EXECS}

clean:
	rm -f ${EXECS}

