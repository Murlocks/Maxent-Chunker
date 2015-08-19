JC = javac
FLAGS = -cp "lib/*;bin;data;."
CLASSES = "com/murlocks/maxent/Chunker"

.SUFFIXES: .java .class

.java.class:
	$(JC) $(FLAGS) $*.java

default: jc

jc javac c compile:
	@test -d bin || mkdir bin
	@$(foreach i, ${CLASSES}, javac -d "bin" ${FLAGS} src/$(i).java;)

run:
	@$(foreach i, ${CLASSES}, java $(FLAGS) $(i);)

clean:
	-rm -rf bin/* *.log *.features *.model
