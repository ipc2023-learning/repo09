file(GLOB SRC_FILES "*.cpp")
add_library(formalism STATIC ${SRC_FILES})
target_link_libraries(formalism libraries)
