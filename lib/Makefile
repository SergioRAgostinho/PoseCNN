TF_INC = $(shell python -c "import tensorflow as tf; print(tf.sysconfig.get_include())")
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_PATH = /usr/local/cuda

I_FLAGS = -I$(TF_INC) -I$(TF_INC)/external/nsync/public
NVCC_FLAGS = -std=c++11 -c $(I_FLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
			 -Xcompiler  -w -DNDEBUG\
			 -arch=sm_50 --expt-relaxed-constexpr \
			 -D _GLIBCXX_USE_CXX11_ABI=0 \
			 $(TF_CFLAGS)
CXX_FLAGS = -std=c++11 -shared $(I_FLAGS) -fPIC -D _GLIBCXX_USE_CXX11_ABI=0 \
			$(TF_CFLAGS)
LD_FLAGS =  -Wl,--no-as-needed $(TF_LFLAGS) -lcudart -lcublas \
            -L $(CUDA_PATH)/lib64 -L$(TF_LIB) -ltensorflow_framework
HV_LFLAGS= -lopencv_imgproc -lopencv_calib3d -lopencv_core -lgomp -lnlopt


all: $(shell echo $(wildcard */*_op_gpu.cu.cc) | sed -r 's/_op_gpu.cu.cc/.so/g') $(shell echo $(wildcard */*_op.cu.cc) | sed -r 's/_op.cu.cc/.so/g') hough_voting_layer/hough_voting.so
	python setup.py build_ext --inplace

clean:
	rm -fv */*.o */*.so

%_gpu.so: %_gpu_op.cu.o %_gpu_op.cc
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) $(HV_LFLAGS) -o $@ $?

%_gpu_op.cu.o: %_gpu_op.cu.cc
	nvcc $(NVCC_FLAGS) -o $@ $?

%.so: %_op.cu.o %_op.cc
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ $?

%.cu.o: %_gpu.cu.cc
	nvcc $(NVCC_FLAGS) -o $@ $?

# Hough Voting

hough_voting_layer/thread_rand.o: hough_voting_layer/thread_rand.cpp
	g++ -std=c++11 -c -fPIC -o $@  $?

hough_voting_layer/Hypothesis.o: hough_voting_layer/Hypothesis.cpp
	g++ -std=c++11 -c -fPIC -o $@  $?

hough_voting_layer/hough_voting.so: hough_voting_layer/hough_voting_op.cc hough_voting_layer/thread_rand.o hough_voting_layer/Hypothesis.o
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) $(HV_LFLAGS) -o $@ $?
