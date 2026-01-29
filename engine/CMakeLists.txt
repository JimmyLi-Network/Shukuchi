add_library(libengine STATIC
    src/engine.c
    src/gguf_reader.c
    src/kv_cache.c
    src/llama_tensor_map.c
    src/model_loader.c
    src/ops.c
    src/prefetch.c
    src/metal_ops.m
)

target_include_directories(libengine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/engine/include
)

find_package(Threads REQUIRED)
target_link_libraries(libengine PRIVATE Threads::Threads)

if(APPLE)
    target_link_libraries(libengine PRIVATE "-framework Metal" "-framework Foundation")
endif()

add_executable(gguf_dump
    tests/gguf_dump.c
)
target_link_libraries(gguf_dump PRIVATE libengine)
target_include_directories(gguf_dump PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
)

add_executable(prefetch_test
    tests/prefetch_test.c
)
target_link_libraries(prefetch_test PRIVATE libengine)
target_include_directories(prefetch_test PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
)

add_executable(kv_cache_test
    tests/kv_cache_test.c
)
target_link_libraries(kv_cache_test PRIVATE libengine)
target_include_directories(kv_cache_test PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
)

add_executable(ops_test
    tests/ops_test.c
)
target_link_libraries(ops_test PRIVATE libengine)
target_include_directories(ops_test PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
)

if(APPLE)
    add_executable(metal_probe
        tests/metal_probe.m
    )
    target_link_libraries(metal_probe PRIVATE "-framework Metal" "-framework Foundation")
endif()
