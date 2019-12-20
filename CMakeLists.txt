CMAKE_MINIMUM_REQUIRED (VERSION 3.13.4)
PROJECT (FFNeuralNetwork)

SET (APPLICATION_CODENAME "${PROJECT_NAME}")

SET (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

SET (PROJECT_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")

SET (CMAKE_CXX_FLAGS " -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11")
SET (CMAKE_C_FLAGS " -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11")
    
if (DEFINED INSTALL_PREFIX)
    message("INSTALLDIR set to " ${INSTALL_PREFIX})
    SET(CMAKE_INSTALL_PREFIX "${INSTALL_PREFIX}")
else (DEFINED INSTALL_PREFIX)
	message("INSTALLDIR set to default" ${CMAKE_INSTALL_PREFIX})
endif()




INCLUDE_DIRECTORIES("${PROJECT_INCLUDE_DIRS}")

add_subdirectory(projects)
