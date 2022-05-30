TARGET       = pricer.x
SRC_DIR      = ./src
INCLUDE_DIR  = ./include
OBJ_DIR      = ./obj
TEST_DIR     = ./tests
FILTRED      = ./src/pay_off.cu ./src/prices_schedule.cu

CC    := nvcc #COMPILER
FLAGS :=-dc

SRCFILES      = $(wildcard $(SRC_DIR)/*.cu) #create the source file list
SRCFILES	 := $(filter-out $(FILTRED), $(SRCFILES))
OBJECTS      := $(SRCFILES:.cu=.o)

SRCTEST      := $(wildcard $(TEST_DIR)/*.cu)
TARGET_TEST  := $(SRCTEST:.cu=.x)
OBJ_TEST     := $(SRCTEST:.cu=.o)

# all: $(TARGET) not available for the moment

test: $(TARGET_TEST);


$(TARGET_TEST): $(OBJECTS) $(OBJ_TEST); 
	\$(CC) -g $(FLAGS) $(OBJECTS) $(subst .x,.o,$@) -o $@ && ./$@
	


$(TARGET): $(OBJECTS);
		$(CC) -g $(FLAGS) $(OBJECTS) -o $(TARGET) \
		mv -f $(SOURCE_DIR)/*.o $(OBJECT_DIR)

%.o: %.cu 
	$(CC) $(FLAGS) -c $< -o $@

clear:
	rm -f $(OBJECTS) $(OBJ_TEST) $(TARGET) $(TARGET_TEST) 


