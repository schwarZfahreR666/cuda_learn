add_executable(hgemm_wmma hgemm_wmma.cu)

#添加行号等信息
find_package(CUDAToolkit REQUIRED)
target_link_libraries(hgemm_wmma PRIVATE CUDA::cudart CUDA::cublas)
set_property(TARGET hgemm_wmma PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(hgemm_wmma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(hgemm_mma hgemm_mma.cu)
target_link_libraries(hgemm_mma PRIVATE CUDA::cudart CUDA::cublas)
set_property(TARGET hgemm_mma PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(hgemm_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(hgemm_bank_conflict hgemm_bank_conflict.cu)
target_link_libraries(hgemm_bank_conflict PRIVATE CUDA::cudart CUDA::cublas)
set_property(TARGET hgemm_bank_conflict PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(hgemm_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)