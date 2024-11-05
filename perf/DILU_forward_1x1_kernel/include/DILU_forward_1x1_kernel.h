#include "amg_config.h"
#include "vector.h"

#ifndef __DILU_forward_1x1_kernel__
#    define __DILU_forward_1x1_kernel__

template< typename Matrix_type, typename Vector_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__MUSA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_1x1_kernel( const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Matrix_type *__restrict A_vals,
                              const int *__restrict A_diag,
                              const Vector_type *x,
                              const Vector_type *b,
                              Vector_type *__restrict delta,
                              const int *__restrict sorted_rows_by_color,
                              const int num_rows_per_color,
                              const int current_color,
                              const int *__restrict row_colors,
                              const Matrix_type *__restrict Einv,
                              const amgx::ColoringType boundary_coloring,
                              const int boundary_index );
#endif /*__DILU_forward_1x1_kernel__*/