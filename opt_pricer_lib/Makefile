TARGET       = pricer.x
SRC_DIR      = ./src
INCLUDE_DIR  = ./include
OBJ_DIR      = ./obj
TEST_DIR     = ./tests
FILTRED      = ./src/pay_off.cu ./src/prices_schedule.cu

CC    := nvcc #COMPILER
LDFLAGS = 
CCFLAGS = -rdc=true 
SRCFILES      = $(wildcard $(SRC_DIR)/*.cu) #create the source file list
SRCFILES	 := $(filter-out $(FILTRED), $(SRCFILES))
OBJECTS      := $(SRCFILES:.cu=.o)

SRCTEST      := $(wildcard $(TEST_DIR)/*.cu)
TARGET_TEST  := $(SRCTEST:.cu=.x)
OBJ_TEST     := $(SRCTEST:.cu=.o)

# all: $(TARGET) not available for the moment

test: $(TARGET_TEST);


$(TARGET_TEST): $(OBJECTS) $(OBJ_TEST); 
	\$(CC) $(LDFLAGS) $(OBJECTS) $(subst .x,.o,$@) -o $@ && ./$@
	


$(TARGET): $(OBJECTS);
		$(CC) $(LDFLAGS) $(OBJECTS) -o $(TARGET) \

%.o: %.cu 
	$(CC) $(CCFLAGS) -c $< -o $@

clear:
	rm -f $(OBJECTS) $(OBJ_TEST) $(TARGET) $(TARGET_TEST) 


