target_sources(mrchem-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fiedler_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/chemtensor_solver.cpp
  )

add_Catch_test(
  NAME fiedler
  LABELS "fiedler;short"
  )

add_Catch_test(
  NAME chemtensor_solver
  LABELS "chemtensor_solver;fiedler"
  )
