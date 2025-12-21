add_executable(transpose transpose.cu)
#添加行号等信息
set_property(TARGET transpose PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo -src-in-ptx>)