CC=g++
CCFLAGS=-Wall -Wpedantic -Werror

main : clean 
	$(CC) -g -std=c++14 main.cpp -I include/ $(CCFLAGS) -o main

clean : 
	rm -f main
