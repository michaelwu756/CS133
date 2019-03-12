SRCS = cnn.h cnn.cpp opencl.cpp main.cpp

REQUIRED_FILES = lab4-report.pdf nvidia.cl params.sh

export OPENCL_PLATFORM= NVIDIA CUDA
export OPENCL_DEVICE= Tesla M60
export OPENCL_SOURCE ?= nvidia.cl

test: cnn $(OPENCL_SOURCE)
	. ./params.sh; ./$<

test-seq:
	@$(MAKE) --no-print-directory test SEQUENTIAL=

test-vadd: vadd vadd.cl
	OPENCL_SOURCE=vadd.cl ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

vadd: opencl.cpp vadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

CXXFLAGS = -std=c++11 -O3 -fopenmp \
	   -fno-loop-interchange -fno-loop-unroll-and-jam -march=native

clang-tidy: $(SRCS)
	clang-tidy $(filter %.cpp, $^) -- -Wall

tar: $(UID).tar.gz

$(UID).tar.gz: $(REQUIRED_FILES)
	@test "$(UID)" -eq "$(UID)" 2>/dev/null || \
	  (echo "Usage: make tar UID=<your 9-digit student id>"; exit 1)
	@$(MAKE)
	@rm -rf $(UID)
	@mkdir $(UID)
	@cp $^ $(UID)
	tar czvf $(UID).tar.gz $(UID)
	@rm -rf $(UID)

check:
	@tar tf $(UID).tar.gz $(addprefix $(UID)/, $(REQUIRED_FILES))

CXXFLAGS := $(subst -march=native,,$(CXXFLAGS))
LDFLAGS += -lOpenCL
