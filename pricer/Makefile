FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(1),$(v)),,$(v)))
FILTER_IN  = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v),))
#Compiler and Linker
CC          := nvcc

#The Target Binary Program
TARGET      := $(basename $(notdir $(wildcard ./src/exam/*.cu )))
TEST        := $(basename $(notdir $(wildcard ./src/tests/*.cu)))
#TEST        := $(basename $(notdir $(wildcard ./src/tests/montecarlo_test2.cu)))
#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
INCDIR      := inc
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cu
DEPEXT      := d
OBJEXT      := o
#Flags, Libraries and Includes
LDFLAGS  = -std=c++11 -Xnvlink --disable-warnings -Wno-deprecated-gpu-targets
CCFLAGS := -rdc=true  -std=c++11  -Wno-deprecated-gpu-targets
INC         := -I$(INCDIR) -I/usr/local/include
INCDEP      := -I$(INCDIR)

SOURCES     := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
FILTER_OBJ  := $(call FILTER_OUT,test, $(OBJECTS))

#Default Make
all: resources $(TEST) $(TARGET) #compile all

#Remake
remake: cleaner all

#exam 
exam: resources $(TARGET)

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
	@$(CC) $(LDFLAGS) -o $(TARGETDIR)/$@ $(FILTER_OBJ) $(call FILTER_IN,$(basename $@),$(OBJECTS)) 

$(TEST): $(OBJECTS)
	@printf "\n"
	@$(CC) $(LDFLAGS) -o $(TARGETDIR)/$@ $(FILTER_OBJ) $(call FILTER_IN,$(basename $@),$(OBJECTS))
	./$(TARGETDIR)/$@ 




#Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c -o $@ $<
	@$(CC) $(CCFLAGS) -M $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

#Non-File Targets
.PHONY: all remake clean cleaner resources


