CXXFLAGS = -std=c++11 -O3 -fopenmp \
					 -fno-loop-interchange -fno-loop-unroll-and-jam -march=native

SRCS = gemm.h gemm.cpp omp.cpp omp-blocked.cpp gemm-baseline.a

test: gemm
	./$^ parallel parallel-blocked

gemm: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a, $^)

clang-tidy: $(SRCS)
	clang-tidy $(filter %.cpp, $^) -- -Wall
	
tar: omp.cpp omp-blocked.cpp lab1-report.pdf
	@test "$(UID)" -eq "$(UID)" 2>/dev/null || \
		(echo "Usage: make tar UID=<your 9-digit student id>"; exit 1)
	@$(MAKE)
	@rm -rf $(UID)
	@mkdir $(UID)
	@cp $^ $(UID)
	tar czvf $(UID).tar.gz $(UID)
	@rm -rf $(UID)
