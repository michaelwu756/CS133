SRCS = cnn.h cnn.cpp opencl.cpp main.cpp

REQUIRED_FILES = lab5-report.pdf xilinx.cl
OPTIONAL_FILES = merlin/src/CnnKernel.cpp

DSA := xilinx_aws-vu9p-f1-04261818_dynamic_5_0
DEVICE := $(AWS_PLATFORM_DYNAMIC_5_0)
XOCC := $(XILINX_SDX)/bin/xocc
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps

export OPENCL_PLATFORM = Xilinx
export OPENCL_DEVICE = $(DSA)
export OPENCL_BINARY = cnn.$(TARGET).$(DSA).xclbin

swsim: emconfig
	XCL_EMULATION_MODE=sw_emu $(MAKE) TARGET=sw_emu dontrunme

hwsim: emconfig
	XCL_EMULATION_MODE=hw_emu $(MAKE) TARGET=hw_emu dontrunme

hls:
	$(MAKE) TARGET=hw cnn.hw.$(DSA).xo

# Building kernel
cnn.$(TARGET).$(DSA).xo: ./xilinx.cl
	$(XOCC) $(CLFLAGS) -c -k CnnKernel -I'$(<D)' -o'$@' '$<'

cnn.$(TARGET).$(DSA).xclbin: cnn.$(TARGET).$(DSA).xo
	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk CnnKernel:1 -o'$@' $(+)

emconfig:emconfig.json

emconfig.json:
	emconfigutil --platform $(DEVICE) --od . 


dontrunme: cnn $(OPENCL_SOURCE) cnn.$(TARGET).$(DSA).xclbin
	. ./params.sh; ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

CXXFLAGS = -std=c++11 -O3 -fopenmp \
           -fno-loop-interchange -fno-loop-unroll-and-jam -march=native
CXXFLAGS := $(subst -fno-loop-unroll-and-jam,,$(subst -march=native,,$(CXXFLAGS)))
LDFLAGS += -lOpenCL

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
