target_sources(mrchem
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/PolyInterpolator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearMaximizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RRMaximizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi_bcast.cpp
  )

add_subdirectory(gto_utils)
