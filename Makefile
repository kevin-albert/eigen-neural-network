CXX		= g++
CXXFLAGS= -Ofast --std=c++14 -msse2
LD 		= g++
LDFLAGS	= -msse2
EXE		= main

main: main.o mnist.o
	$(LD)  $^ -o $@

clean: 
	rm -f *.o $(EXE)
