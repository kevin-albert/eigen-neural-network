CXX 	= g++-6
CXXFLAGS= -Ofast --std=c++17 -msse2 -fopenmp -march=native \
 		  -I../../src/ff -I../../lib
LDFLAGS	=
LD 		= g++-6 -fopenmp
EXE		= main

$(EXE): $(EXE).o
	$(LD)  $^ -o $@

$(EXE).o: mnist.h hack.h network.h ../../src/ff/nn.h ../../src/ff/nn.hpp

clean: 
	rm -f *.o $(EXE)
