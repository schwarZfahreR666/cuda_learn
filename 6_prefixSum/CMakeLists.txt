add_executable(prefix_sum prefix_sum.cu)
#添加行号等信息
set_property(TARGET prefix_sum PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(prefix_sum PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
