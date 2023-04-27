file(GLOB SRC_FILES "*.cpp")
add_library(dataset STATIC ${SRC_FILES})
target_link_libraries(dataset planners)
