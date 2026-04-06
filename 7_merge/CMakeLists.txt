add_executable(merge merge.cu)
#添加行号等信息
set_property(TARGET merge PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
