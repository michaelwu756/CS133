CXX = mpicxx

SRCS = ../lab1/gemm.h ../lab1/gemm.cpp ../lab1/gemm-baseline.a \
       mpi.cpp main.cpp

REQUIRED_FILES = mpi.cpp lab2-report.pdf

np ?= 4

test: gemm
	mpiexec -np $(np) ./$^

gemm: $(SRCS)
	$(CXX) $(subst -fopenmp,,$(CXXFLAGS)) -o $@ $(filter %.cpp %.a %.o, $^) -lgomp

../lab1/gemm-baseline.a:
	$(MAKE) -C ../lab1 gemm-baseline.a

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

check: $(UID).tar.gz
	@tar tf $^ $(addprefix $(UID)/, $(REQUIRED_FILES))
