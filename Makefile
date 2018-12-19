

CUDNN_PATH = /home/scratch.svc_compute_arch/release/cudnn/v7.3_cuda_10.0/latest
CUDA_PATH  = /home/scratch.svc_compute_arch/release/cuda_toolkit/r10.0/latest

GENCODE_FLAGS := \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-Xptxas="--ext-desc-file $(CUDA_PATH)/bin/ptxExtDesc.txt"

DEBUG_FLAGS := -g -lineinfo

NVCC_FLAGS := $(GENCODE_FLAGS) $(DEBUG_FLAGS) -O3 -std=c++11 -m64 --use_fast_math -Xptxas '-v'

INCLUDES  := -I$(CUDNN_PATH)/include -I$(CUDA_PATH)/include -I. -I/home/utils/boost-1.64.0/gcc-5.3.0/include
LIBRARIES := -lcudnn -lstdc++fs -L$(CUDNN_PATH)/lib64


NVCC      := $(CUDA_PATH)/bin/nvcc

draw_clover_SOURCES := clover_kernel.cu main.cpp
draw_clover_OBJECTS := $(patsubst %.cu, obj/%.cu.o, $(patsubst %.cpp, obj/%.o, $(test_memcpy_SOURCES)))
ALL_TARGETS := draw_clover

obj/%.cu.o: %.cu obj/%.cu.d | obj
	@ echo "Compiling Cuda $@"
	$(NVCC) -Xcompiler='-MD -MP' \
		$(INCLUDES) $(NVCC_FLAGS) \
		$< -dc -x cu -o $@ 
	@ mv $(patsubst %.cu, %.d, $(notdir $<)) $(@:.o=.d) 
	@ sed -i 's/$(patsubst %.cu, %.o, $(notdir $<))/$(subst /,\/,$@)/' $(@:.o=.d) && touch $@

obj/%.o: %.cpp obj/%.d | obj
	@ echo "Compiling CXX $@"
	$(NVCC) -Xcompiler='-MT $@ -MMD -MP -MF obj/$*.d.tmp'		 	\
		$(INCLUDES) $(NVCC_FLAGS)	\
		-c $< -o $@
	@ mv -f obj/$*.d.tmp obj/$*.d && touch $@

obj:
	mkdir -p $@

# empty rule for dependency include file
obj/%.cu.d: ;


obj/%.d: ;


-include $(test_memcpy_OBJECTS:.o=.d)
-include $(test_conv_OBJECTS:.o=.d)

clean:
	rm -f obj/*.o *.cubin* *.i *.ii *.cudafe* *.fatbin* *.ptx *.sass *.exe
