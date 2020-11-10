include_directories (${PROJECT_INCLUDE_DIRS})



add_executable(NN main_with_eigen.cpp)

install(TARGETS NN DESTINATION lib)


if(WITHOUTEIGEN) 
   add_executable(NN_WE main_without_eigen.cpp)
   install(TARGETS NN_WE DESTINATION lib)
endif()
