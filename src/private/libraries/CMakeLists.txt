file(GLOB SRC_FILES "*.cpp")
file(GLOB SRC_MICROTAR_FILES "microtar/*.c")
add_library(libraries STATIC ${SRC_FILES} ${SRC_MICROTAR_FILES} ${SRC_CLIQUES_FILES})
