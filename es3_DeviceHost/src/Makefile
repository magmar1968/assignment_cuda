SRC= myRandom.cu main.cu  #source files
OBJECTS=$(SRC:.cu=.o)     #file compilati da linkare
CCFLAG= --std=c++03 -dc                 #eventuali flag
LDFLAG= --std=c++03
TARGET=../main.x          #nome del file da compilare
CC=nvcc                   #compilatore


# Nel makefile si definiscono dei comandi a cui si associano 
# delle funzioni.
# in bash per aver accesso ad una variabile si usa il dollaro 
# quindi la linea seguente dice "compila ciò che è contenuto
# nella variabile target 
all: $(TARGET)        

# la struttura di un comando è: 
# nomecomando: cosa mi serve
# 	tab o ; cosa devo fare
$(TARGET): $(OBJECTS)
	$(CC) -g $(LDFLAG) $(OBJECTS) -o $@

%.o: %.cu 
	$(CC) $(CCFLAG) -c $< -o $@
main.o: main.cu
	$(CC) $(CCFLAG) -c $< -o $@

clear:
	rm *.o $(TARGET)
# questo makefile è leggermente pià arzigogolato perchè usa delle
# variabili predefinite il cui significato si può ritrovare a questa
# pagina http://labmaster.mi.infn.it/Laboratorio2/serale/l1/index.html
# In generale comunque nessuno scrive più Makefile a mano, ma solitamente
# lo si fa fare al proprio editor di fiducia o si usano programmi come 
# Cmake o simili.



