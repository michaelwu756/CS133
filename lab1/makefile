SRCS = gemm.h gemm.cpp main.cpp omp.cpp omp-blocked.cpp gemm-baseline.a

REQUIRED_FILES = omp.cpp omp-blocked.cpp lab1-report.pdf

test: gemm
	./$^ parallel parallel-blocked

gemm: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a, $^)

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

