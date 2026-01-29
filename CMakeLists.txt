cmake_minimum_required(VERSION 3.16)

project(shukuchi C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()

add_subdirectory(engine)
add_subdirectory(model_packer)

add_executable(shukuchi engine/src/main.c)
target_include_directories(shukuchi PRIVATE include engine/include)
target_link_libraries(shukuchi PRIVATE libengine)
