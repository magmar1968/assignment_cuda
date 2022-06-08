FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(1),$(v)),,$(v)))
FILTER_IN  = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v),))
#Compiler and Linker
CC          := nvcc

#The Target Binary Program
TARGET      := #pricer
TEST        := $(basename $(notdir $(wildcard ./src/tests/*.cu)))
#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
INCDIR      := inc
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cu
DEPEXT      := d
OBJEXT      := o

#Flags, Libraries and Includes
LDFLAGS  = --gpu-architecture=sm_50
CCFLAGS := -rdc=true --gpu-architecture=sm_50
# INC         := -I$(INCDIR) -I/usr/local/include
# INCDEP      := -I$(INCDIR)

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
SOURCES     := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
FILTER_OBJ  := $(call FILTER_OUT,test, $(OBJECTS))

#Defauilt Make
all: resources $(TEST) #$(TARGET) #compile all

#Remake
remake: cleaner all

#Copy Resources from Resources Directory to Target Directory
resources: directories
    # @cp $(RESDIR)/* $(TARGETDIR)/

#Make the Directories
directories:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#Clean only Objecst
clean:
	@$(RM) -r $(BUILDDIR)

#Full Clean, Objects and Binaries
cleaner: clean
	@$(RM) -r $(TARGETDIR)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))

#Link
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $(TARGETDIR)/$@ $^ $(LIB)

$(TEST): $(OBJECTS)
	@printf "\n"
	$(CC) $(LDFLAGS) -o $(TARGETDIR)/$@ $(FILTER_OBJ) $(call FILTER_IN,$(basename $@),$(OBJECTS))\
	&& ./$(TARGETDIR)/$@

#Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c -o $@ $<
	@$(CC) $(CCFLAGS) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

#Non-File Targets
.PHONY: all remake clean cleaner resources







































# TARGET       = pricer.x
# SRC_DIR      = ./src
# LIB_DIR      = ./lib



# INCLUDE_DIR  = ./include
# OBJ_DIR      = ./obj
# TEST_DIR     = ./tests
# FILTRED      = ./src/pay_off.cu ./src/prices_schedule.cu

# CC    := nvcc #COMPILER
# LDFLAGS = 
# CCFLAGS = -rdc=true --gpu-architecture=sm_20

# LIBS = $(wildcard $(LIB_DIR)/*/*.cu)
# LIBS += $(wildcard $(LIB_DIR)/*/*/*.cu)

# print:
# 	$(LIBS)


# SRCFILES      = $(wildcard $(SRC_DIR)/*.cu) #create the source file list
# SRCFILES	 := $(filter-out $(FILTRED), $(SRCFILES))
# OBJECTS      := $(SRCFILES:.cu=.o)

# SRCTEST      := $(wildcard $(TEST_DIR)/*.cu)
# TARGET_TEST  := $(SRCTEST:.cu=.x)
# OBJ_TEST     := $(SRCTEST:.cu=.o)

# # all: $(TARGET) not available for the moment

# test: $(TARGET_TEST);


# $(TARGET_TEST): $(OBJECTS) $(OBJ_TEST); 
# 	\$(CC) $(LDFLAGS) $(OBJECTS) $(subst .x,.o,$@) -o $@ 
	


# $(TARGET): $(OBJECTS);
# 		$(CC) $(LDFLAGS) $(OBJECTS) -o $(TARGET) \

# %.o: %.cu 
# 	$(CC) $(CCFLAGS) -c $< -o $@
# 	@cp $*.d $*.P; \
#             sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
#                 -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
#             rm -f $*.d

# -include $(OBJECTS:%.o=%.P)

# clear:
# 	rm -f $(OBJECTS) $(OBJ_TEST) $(TARGET) $(TARGET_TEST) 


