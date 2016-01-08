cmake_minimum_required(VERSION 3.0)

project(warg-cv-bootcamp CXX) 

set(MODULES)

# Dependencies (none)

enable_testing()

# Include Modules

foreach(p ${MODULES})
    add_subdirectory("modules/${p}")
endforeach()

foreach(p ${MODULES})
    include_directories("modules/${p}/include")
endforeach()

if (UNIX)
    message(status "Setting GCC flags")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fexceptions -Wall -Wextra")
else()
    message(status "Setting MSVC flags")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHc-")
endif()
message(status " CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(status " CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
