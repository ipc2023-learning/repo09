file(GLOB SRC_FILES "*.cpp")
add_library(algorithms STATIC ${SRC_FILES})
target_link_libraries(algorithms libraries)
