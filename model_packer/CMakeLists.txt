add_executable(model_packer
    src/main.c
    src/packer.c
)

target_include_directories(model_packer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/model_packer/include
)
