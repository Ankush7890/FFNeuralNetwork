include_directories (${PROJECT_INCLUDE_DIRS})


add_executable(NN main.cpp)
install(TARGETS NN DESTINATION lib)
