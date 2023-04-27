add_subdirectory(public)
add_subdirectory(private)

# -----
# TRAIN
# -----
add_executable(train train.cpp helpers.cpp train_dl.cpp train_rl.cpp train_ol.cpp)
set_property(TARGET train PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(train PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(train -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(train ${TORCH_LIBRARIES})
target_link_libraries(train dataset)
target_link_libraries(train experiments)
target_link_libraries(train formalism)
target_link_libraries(train model)
target_link_libraries(train pddl)
target_link_libraries(train planners)

# ---------
# IPC TRAIN
# ---------
add_executable(ipc_train ipc_train.cpp helpers.cpp)
set_property(TARGET ipc_train PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(ipc_train PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(ipc_train -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(ipc_train ${TORCH_LIBRARIES})
target_link_libraries(ipc_train dataset)
target_link_libraries(ipc_train experiments)
target_link_libraries(ipc_train formalism)
target_link_libraries(ipc_train model)
target_link_libraries(ipc_train pddl)
target_link_libraries(ipc_train planners)

# ----
# PLAN
# ----
add_executable(plan plan.cpp)
set_property(TARGET plan PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(plan PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(plan -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(plan ${TORCH_LIBRARIES})
target_link_libraries(plan formalism)
target_link_libraries(plan model)
target_link_libraries(plan pddl)
target_link_libraries(plan planners)

# --------
# IPC PLAN
# --------
add_executable(ipc_plan ipc_plan.cpp)
set_property(TARGET ipc_plan PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(ipc_plan PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(ipc_plan -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(ipc_plan ${TORCH_LIBRARIES})
target_link_libraries(ipc_plan formalism)
target_link_libraries(ipc_plan model)
target_link_libraries(ipc_plan pddl)
target_link_libraries(ipc_plan planners)

# ------
# SEARCH
# ------
add_executable(search search.cpp)
set_property(TARGET search PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(search PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(search -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(search algorithms)
target_link_libraries(search formalism)
target_link_libraries(search pddl)
target_link_libraries(search planners)

# -------------
# APPROXIMATION
# -------------
add_executable(approximation approximation.cpp)
set_property(TARGET approximation PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(approximation PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(approximation -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(approximation algorithms)
target_link_libraries(approximation formalism)
target_link_libraries(approximation pddl)
target_link_libraries(approximation planners)

# -----------
# STATE SPACE
# -----------
add_executable(state_space state_space.cpp)
set_property(TARGET state_space PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(state_space PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(state_space -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(state_space formalism)
target_link_libraries(state_space pddl)
target_link_libraries(state_space planners)

# -------
# CLASSES
# -------
add_executable(classes classes.cpp)
set_property(TARGET classes PROPERTY CXX_STANDARD 17)

if(MSVC)
    set_target_properties(classes PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
endif(MSVC)

target_link_libraries(classes -static-libstdc++ -lstdc++fs -static-libgcc) # Makes the executable more portable
target_link_libraries(classes algorithms)
target_link_libraries(classes formalism)
target_link_libraries(classes pddl)
target_link_libraries(classes planners)

if(MSVC)
    add_custom_command(TARGET train
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:train>)

    add_custom_command(TARGET plan
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:plan>)

        add_custom_command(TARGET ipc_train
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:ipc_train>)

    add_custom_command(TARGET ipc_plan
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:ipc_plan>)
endif(MSVC)
