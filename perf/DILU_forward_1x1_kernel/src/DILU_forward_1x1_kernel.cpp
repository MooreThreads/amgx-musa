#include "DILU_forward_1x1_kernel.h"
#include <string.h>
#include <cutil.h>
#include <miscmath.h>
#include <amgx_cusparse.h>
#include <thrust/copy.h>
#include <solvers/multicolor_dilu_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>
#include <ld_functions.h>
#include <matrix_io.h>
#include <thrust/logical.h>
#include <profile.h>
#include <sm_utils.inl>
#include <amgx_types/util.h>
#include <algorithm>

template< typename Matrix_type, typename Vector_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE, bool HAS_EXTERNAL_DIAG >
__global__
__launch_bounds__( CTA_SIZE, 12 )
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
                              const int boundary_index )
{
    // Number of items per CTA.
    const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
    // Number of items per grid.
    const int NUM_ROWS_PER_GRID = gridDim.x * NUM_ROWS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_mod_NTPR = lane_id % NUM_THREADS_PER_ROW;
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ROWS_PER_CTA + (threadIdx.x / NUM_THREADS_PER_ROW);

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ROWS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx = amgx::types::util<Vector_type>::get_zero();

        if ( lane_id_mod_NTPR == 0 )
        {
            my_bmAx = amgx::__cachingLoad(&b[a_row_id]);
        }

        // If it has an external diag.
        if ( HAS_EXTERNAL_DIAG && lane_id_mod_NTPR == 0 )
        {
            my_bmAx -= A_vals[A_diag[a_row_id]] * x[a_row_id];
        }

        // Don't do anything if X is zero.
        int a_col_it  = A_rows[a_row_id  ];
        int a_col_end = A_rows[a_row_id + 1];

        // If the diagonal is stored separately, we have a special treatment.
        //if( HAS_EXTERNAL_DIAG )
        //  ++a_col_end;

        // Each warp load column indices of 32 nonzero blocks
        for ( a_col_it += lane_id_mod_NTPR ; utils::any( a_col_it < a_col_end ) ; a_col_it += NUM_THREADS_PER_ROW )
        {
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Ignore the diagonal element since its color is smaller, and been accounted for above
            if (HAS_EXTERNAL_DIAG && a_col_id == a_row_id)
            {
                a_col_id = -1;
            }

            // Load x.
            Vector_type my_x(0);

            if ( a_col_id != -1 )
            {
                my_x = amgx::__cachingLoad(&x[a_col_id]);
            }

            // Is it really a valid column (due to coloring).
            int valid = false;

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == amgx::FIRST )
                {
                    valid = a_col_id >= boundary_index;
                }
                else
                {
                    valid = a_col_id < boundary_index && row_colors[a_col_id] < current_color;
                }
            }
            // Load my x value.
            if ( valid )
            {
                my_x += delta[a_col_id];
            }

            // Load my item from A.
            Matrix_type my_val(0);

            if ( a_col_it < a_col_end )
            {
                my_val = A_vals[a_col_it];
            }

            // Update bmAx.
            my_bmAx -= my_val * my_x;
        }

        // Reduce bmAx terms.
#pragma unroll

        for ( int mask = NUM_THREADS_PER_ROW / 2 ; mask > 0 ; mask >>= 1 )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, mask );
        }

        // Store the results.
        if ( lane_id_mod_NTPR == 0 )
        {
            delta[a_row_id] = Einv[a_row_id] * my_bmAx;
        }
    }
}


template __global__ void DILU_forward_1x1_kernel<float, float, 8, 128, 32, false>(
    const int* __restrict A_rows,
    const int* __restrict A_cols,
    const float* __restrict A_vals,
    const int* __restrict A_diag,
    const float* x,
    const float* b,
    float* __restrict delta,
    const int* __restrict sorted_rows_by_color,
    const int num_rows_per_color,
    const int current_color,
    const int* __restrict row_colors,
    const float* __restrict Einv,
    const amgx::ColoringType boundary_coloring,
    const int boundary_index);
