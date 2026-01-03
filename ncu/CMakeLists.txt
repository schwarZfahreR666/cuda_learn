add_executable(combine_memory_access combine_memory_access.cu)
#添加行号等信息
set_property(TARGET combine_memory_access PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(combine_memory_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
