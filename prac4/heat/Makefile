CC = icx

NAME = heat

OPTRPT = heat-openmp-spir64

default:
	$(CC) -fiopenmp -fopenmp-targets=spir64 -qopt-report=1 $(NAME).c -o $(NAME)

env:


run:
	LIBOMPTARGET_PLUGIN=OPENCL ./$(NAME)

clean:
	rm $(NAME) $(OPTRPT).optrpt $(NAME).optrpt
