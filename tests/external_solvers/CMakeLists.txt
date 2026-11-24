target_sources(mrchem-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fiedler_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/chemtensor_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi_bcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lagrangian_update.cpp
  )

add_Catch_test(
  NAME fiedler
  LABELS "fiedler;short"
  )

add_Catch_test(
  NAME chemtensor_solver
  LABELS "chemtensor_solver;fiedler"
  )

add_Catch_test(
  NAME mpi_bcast
  LABELS "mpi_bcast;short"
  )

add_Catch_test(
  NAME lagrangian_update
  LABELS "lagrangian_update;short"
  )
