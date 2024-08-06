/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <amgx_cublas.h>
#ifdef AMGX_USE_LAPACK
#include "mkl.h"
#endif
namespace amgx
{

mublasHandle_t Cublas::m_handle = 0;

namespace
{
// real valued calls
mublasStatus_t cublas_axpy(mublasHandle_t handle, int n,
                           const float *alpha,
                           const float *x, int incx,
                           float *y, int incy)
{
    return mublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

mublasStatus_t cublas_axpy(mublasHandle_t handle, int n,
                           const double *alpha,
                           const double *x, int incx,
                           double *y, int incy)
{
    return mublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

mublasStatus_t cublas_copy(mublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *y, int incy)
{
    return mublasScopy(handle, n, x, incx, y, incy);
}

mublasStatus_t cublas_copy(mublasHandle_t handle, int n,
                           const double *x, int incx,
                           double *y, int incy)
{
    return mublasDcopy(handle, n, x, incx, y, incy);
}

mublasStatus_t cublas_dot(mublasHandle_t handle, int n,
                          const float *x, int incx, const float *y, int incy,
                          float *result)
{
    return mublasSdot(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dot(mublasHandle_t handle, int n,
                          const double *x, int incx, const double *y, int incy,
                          double *result)
{
    return mublasDdot(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dotc(mublasHandle_t handle, int n,
                           const float *x, int incx, const float *y, int incy,
                           float *result)
{
    return mublasSdot(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dotc(mublasHandle_t handle, int n,
                           const double *x, int incx, const double *y, int incy,
                           double *result)
{
    return mublasDdot(handle, n, x, incx, y, incy, result);
}


mublasStatus_t cublas_trsv_v2(mublasHandle_t handle,
                              mublasFillMode_t uplo,
                              mublasOperation_t trans,
                              mublasDiagType_t diag,
                              int n,
                              const float *A,
                              int lda,
                              float *x,
                              int incx)
{
    return mublasStrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}
mublasStatus_t cublas_trsv_v2(mublasHandle_t handle,
                              mublasFillMode_t uplo,
                              mublasOperation_t trans,
                              mublasDiagType_t diag,
                              int n,
                              const double *A,
                              int lda,
                              double *x,
                              int incx)
{
    return mublasDtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}

mublasStatus_t cublas_gemm(mublasHandle_t handle,
                           mublasOperation_t transa, mublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
{
    return mublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

mublasStatus_t cublas_gemm(mublasHandle_t handle,
                           mublasOperation_t transa, mublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
{
    return mublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

mublasStatus_t cublas_gemv(mublasHandle_t handle, mublasOperation_t trans, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *x, int incx,
                           const float *beta, float *y, int incy)
{
    return mublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

mublasStatus_t cublas_gemv(mublasHandle_t handle, mublasOperation_t trans, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *x, int incx,
                           const double *beta, double *y, int incy)
{
    return mublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

mublasStatus_t cublas_ger(mublasHandle_t handle, int m, int n,
                          const float *alpha,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *A, int lda)
{
    return mublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_ger(mublasHandle_t handle, int m, int n,
                          const double *alpha,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *A, int lda)
{
    return mublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
mublasStatus_t cublas_gerc(mublasHandle_t handle, int m, int n,
                           const float *alpha,
                           const float *x, int incx,
                           const float *y, int incy,
                           float *A, int lda)
{
    return mublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_gerc(mublasHandle_t handle, int m, int n,
                           const double *alpha,
                           const double *x, int incx,
                           const double *y, int incy,
                           double *A, int lda)
{
    return mublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_nrm2(mublasHandle_t handle, int n,
                           const float *x, int incx, float *result)
{
    return mublasSnrm2(handle, n, x, incx, result);
}

mublasStatus_t cublas_nrm2(mublasHandle_t handle, int n,
                           const double *x, int incx, double *result)
{
    return mublasDnrm2(handle, n, x, incx, result);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const float *alpha,
                           float *x, int incx)
{
    return mublasSscal(handle, n, alpha, x, incx);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const double *alpha,
                           double *x, int incx)
{
    return mublasDscal(handle, n, alpha, x, incx);
}


// complex valued calls
mublasStatus_t cublas_axpy(mublasHandle_t handle, int n,
                           const muComplex *alpha,
                           const muComplex *x, int incx,
                           muComplex *y, int incy)
{
    return mublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

mublasStatus_t cublas_axpy(mublasHandle_t handle, int n,
                           const muDoubleComplex *alpha,
                           const muDoubleComplex *x, int incx,
                           muDoubleComplex *y, int incy)
{
    return mublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

mublasStatus_t cublas_copy(mublasHandle_t handle, int n,
                           const muComplex *x, int incx,
                           muComplex *y, int incy)
{
    return mublasCcopy(handle, n, x, incx, y, incy);
}

mublasStatus_t cublas_copy(mublasHandle_t handle, int n,
                           const muDoubleComplex *x, int incx,
                           muDoubleComplex *y, int incy)
{
    return mublasZcopy(handle, n, x, incx, y, incy);
}

mublasStatus_t cublas_dot(mublasHandle_t handle, int n,
                          const muComplex *x, int incx, const muComplex *y, int incy,
                          muComplex *result)
{
    return mublasCdotu(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dot(mublasHandle_t handle, int n,
                          const muDoubleComplex *x, int incx, const muDoubleComplex *y, int incy,
                          muDoubleComplex *result)
{
    return mublasZdotu(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dotc(mublasHandle_t handle, int n,
                           const muComplex *x, int incx, const muComplex *y, int incy,
                           muComplex *result)
{
    return mublasCdotc(handle, n, x, incx, y, incy, result);
}

mublasStatus_t cublas_dotc(mublasHandle_t handle, int n,
                           const muDoubleComplex *x, int incx, const muDoubleComplex *y, int incy,
                           muDoubleComplex *result)
{
    return mublasZdotc(handle, n, x, incx, y, incy, result);
}


mublasStatus_t cublas_trsv_v2(mublasHandle_t handle,
                              mublasFillMode_t uplo,
                              mublasOperation_t trans,
                              mublasDiagType_t diag,
                              int n,
                              const muComplex *A,
                              int lda,
                              muComplex *x,
                              int incx)
{
    return mublasCtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}
mublasStatus_t cublas_trsv_v2(mublasHandle_t handle,
                              mublasFillMode_t uplo,
                              mublasOperation_t trans,
                              mublasDiagType_t diag,
                              int n,
                              const muDoubleComplex *A,
                              int lda,
                              muDoubleComplex *x,
                              int incx)
{
    return mublasZtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}

mublasStatus_t cublas_gemm(mublasHandle_t handle,
                           mublasOperation_t transa, mublasOperation_t transb,
                           int m, int n, int k,
                           const muComplex           *alpha,
                           const muComplex           *A, int lda,
                           const muComplex           *B, int ldb,
                           const muComplex           *beta,
                           muComplex           *C, int ldc)
{
    return mublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

mublasStatus_t cublas_gemm(mublasHandle_t handle,
                           mublasOperation_t transa, mublasOperation_t transb,
                           int m, int n, int k,
                           const muDoubleComplex          *alpha,
                           const muDoubleComplex          *A, int lda,
                           const muDoubleComplex          *B, int ldb,
                           const muDoubleComplex          *beta,
                           muDoubleComplex          *C, int ldc)
{
    return mublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

mublasStatus_t cublas_gemv(mublasHandle_t handle, mublasOperation_t trans, int m, int n,
                           const muComplex *alpha, const muComplex *A, int lda,
                           const muComplex *x, int incx,
                           const muComplex *beta, muComplex *y, int incy)
{
    return mublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

mublasStatus_t cublas_gemv(mublasHandle_t handle, mublasOperation_t trans, int m, int n,
                           const muDoubleComplex *alpha, const muDoubleComplex *A, int lda,
                           const muDoubleComplex *x, int incx,
                           const muDoubleComplex *beta, muDoubleComplex *y, int incy)
{
    return mublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

mublasStatus_t cublas_ger(mublasHandle_t handle, int m, int n,
                          const muComplex *alpha,
                          const muComplex *x, int incx,
                          const muComplex *y, int incy,
                          muComplex *A, int lda)
{
    return mublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_ger(mublasHandle_t handle, int m, int n,
                          const muDoubleComplex *alpha,
                          const muDoubleComplex *x, int incx,
                          const muDoubleComplex *y, int incy,
                          muDoubleComplex *A, int lda)
{
    return mublasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
mublasStatus_t cublas_gerc(mublasHandle_t handle, int m, int n,
                           const muComplex *alpha,
                           const muComplex *x, int incx,
                           const muComplex *y, int incy,
                           muComplex *A, int lda)
{
    return mublasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_gerc(mublasHandle_t handle, int m, int n,
                           const muDoubleComplex *alpha,
                           const muDoubleComplex *x, int incx,
                           const muDoubleComplex *y, int incy,
                           muDoubleComplex *A, int lda)
{
    return mublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

mublasStatus_t cublas_nrm2(mublasHandle_t handle, int n,
                           const muComplex *x, int incx, float *result)
{
    return mublasScnrm2(handle, n, x, incx, result);
}

mublasStatus_t cublas_nrm2(mublasHandle_t handle, int n,
                           const muDoubleComplex *x, int incx, double *result)
{
    return mublasDznrm2(handle, n, x, incx, result);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const muComplex *alpha,
                           muComplex *x, int incx)
{
    return mublasCscal(handle, n, alpha, x, incx);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const muDoubleComplex *alpha,
                           muDoubleComplex *x, int incx)
{
    return mublasZscal(handle, n, alpha, x, incx);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const float *alpha,
                           muComplex *x, int incx)
{
    return mublasCsscal(handle, n, alpha, x, incx);
}

mublasStatus_t cublas_scal(mublasHandle_t handle, int n,
                           const double *alpha,
                           muDoubleComplex *x, int incx)
{
    return mublasZdscal(handle, n, alpha, x, incx);
}

} // anonymous namespace.

void Cublas::set_pointer_mode_device()
{
    mublasHandle_t handle = Cublas::get_handle();
    mublasSetPointerMode(handle, MUBLAS_POINTER_MODE_DEVICE);
}

void Cublas::set_pointer_mode_host()
{
    mublasHandle_t handle = Cublas::get_handle();
    mublasSetPointerMode(handle, MUBLAS_POINTER_MODE_HOST);
}

template <class TConfig>
void Cublas::gemm(typename TConfig::VecPrec alpha,
                  const Vector<TConfig> &A, const Vector<TConfig> &B,
                  typename TConfig::VecPrec beta, Vector<TConfig> &C,
                  bool A_transposed, bool B_transposed)
{
    mublasOperation_t trans_A = A_transposed ? MUBLAS_OP_T : MUBLAS_OP_N;
    mublasOperation_t trans_B = B_transposed ? MUBLAS_OP_T : MUBLAS_OP_N;
    int m = A_transposed ? A.get_num_cols() : A.get_num_rows();
    int n = B_transposed ? B.get_num_rows() : B.get_num_cols();
    int k = A_transposed ? A.get_num_rows() : A.get_num_cols();
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_gemm(handle, trans_A, trans_B,
                                 m, n, k,
                                 &alpha, A.raw(), A.get_lda(),
                                 B.raw(), B.get_lda(),
                                 &beta, C.raw(), C.get_lda()));
    C.dirtybit = 1;
}

template <typename T>
void Cublas::axpy(int n, T alpha,
                  const T *x, int incx,
                  T *y, int incy)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename T>
void Cublas::copy(int n, const T *x, int incx,
                  T *y, int incy)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_copy(handle, n, x, incx, y, incy));
}

template <typename T>
void Cublas::dot(int n, const T *x, int incx,
                 const T *y, int incy,
                 T *result)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_dot(handle, n, x, incx, y, incy, result));
}

template <typename T>
void Cublas::dotc(int n, const T *x, int incx,
                  const T *y, int incy,
                  T *result)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_dotc(handle, n, x, incx, y, incy, result));
}

template <typename T, typename V>
V Cublas::nrm2(int n, const T *x, int incx)
{
    mublasHandle_t handle = Cublas::get_handle();
    V result;
    Cublas::nrm2(n, x, incx, &result);
    return result;
}

template <typename T, typename V>
void Cublas::nrm2(int n, const T *x, int incx, V *result)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_nrm2(handle, n, x, incx, result));
}

template <typename T, typename V>
void Cublas::scal(int n, T alpha, V *x, int incx)
{
    Cublas::scal(n, &alpha, x, incx);
}

template <typename T, typename V>
void Cublas::scal(int n, T *alpha, V *x, int incx)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_scal(handle, n, alpha, x, incx));
}

template <typename T>
void Cublas::gemv(bool transposed, int m, int n,
                  const T *alpha, const T *A, int lda,
                  const T *x, int incx,
                  const T *beta, T *y, int incy)
{
    mublasHandle_t handle = Cublas::get_handle();
    mublasOperation_t trans = transposed ? MUBLAS_OP_T : MUBLAS_OP_N;
    cublasCheckError(cublas_gemv(handle, trans, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy));
}

template <typename T>
void Cublas::gemv_ext(bool transposed, const int m, const int n,
                      const T *alpha, const T *A, const int lda,
                      const T *x, const int incx,
                      const T *beta, T *y, const int incy, const int offsetx, const int offsety, const int offseta)
{
    mublasHandle_t handle = Cublas::get_handle();
    mublasOperation_t trans = transposed ? MUBLAS_OP_T : MUBLAS_OP_N;
    cublasCheckError(cublas_gemv(handle, trans, m, n, alpha, A + offseta, lda,
                                 x + offsetx, incx, beta, y + offsety, incy));
}

template <typename T>
void Cublas::trsv_v2( mublasFillMode_t uplo, mublasOperation_t trans, mublasDiagType_t diag, int n,
                      const T *A, int lda, T *x, int incx, int offseta)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError( cublas_trsv_v2(handle, uplo, trans, diag, n, A + offseta, lda, x, incx));
}


template <typename T>
void Cublas::ger(int m, int n, const T *alpha,
                 const T *x, int incx,
                 const T *y, int incy,
                 T *A, int lda)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

template <typename T>
void Cublas::gerc(int m, int n, const T *alpha,
                  const T *x, int incx,
                  const T *y, int incy,
                  T *A, int lda)
{
    mublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_gerc(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

#define AMGX_CASE_LINE(CASE) \
    template void Cublas::gemm(typename TemplateMode<CASE>::Type::VecPrec, const Vector<TemplateMode<CASE>::Type>&, const Vector<TemplateMode<CASE>::Type>&, typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>&, bool, bool);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// real valued instantiaions
template void Cublas::axpy(int n, float alpha,
                           const float *x, int incx,
                           float *y, int incy);
template void Cublas::axpy(int n, double alpha,
                           const double *x, int incx,
                           double *y, int incy);

template void Cublas::copy(int n, const float *x, int incx, float *y, int incy);
template void Cublas::copy(int n, const double *x, int incx, double *y, int incy);

template void Cublas::dot(int n, const float *x, int incx,
                          const float *y, int incy,
                          float *result);
template void Cublas::dot(int n, const double *x, int incx,
                          const double *y, int incy,
                          double *result);
template void Cublas::dotc(int n, const float *x, int incx,
                           const float *y, int incy,
                           float *result);
template void Cublas::dotc(int n, const double *x, int incx,
                           const double *y, int incy,
                           double *result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *x, int incx,
                           const float *beta, float *y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *x, int incx,
                           const double *beta, double *y, int incy);

template void Cublas::ger(int m, int n, const float *alpha,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *A, int lda);
template void Cublas::ger(int m, int n, const double *alpha,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *A, int lda);
template void Cublas::gerc(int m, int n, const float *alpha,
                           const float *x, int incx,
                           const float *y, int incy,
                           float *A, int lda);
template void Cublas::gerc(int m, int n, const double *alpha,
                           const double *x, int incx,
                           const double *y, int incy,
                           double *A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const float *alpha, const float *A, const int lda,
                               const float *x, const int incx,
                               const float *beta, float *y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const double *alpha, const double *A, const int lda,
                               const double *x, const int incx,
                               const double *beta, double *y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( mublasFillMode_t uplo, mublasOperation_t trans, mublasDiagType_t diag, int n,
                               const float *A, int lda, float *x, int incx, int offseta);
template void Cublas::trsv_v2( mublasFillMode_t uplo, mublasOperation_t trans, mublasDiagType_t diag, int n,
                               const double *A, int lda, double *x, int incx, int offseta);

template double Cublas::nrm2(int n, const double *x, int incx);
template float Cublas::nrm2(int n, const float *x, int incx);

template void Cublas::scal(int n, float alpha, float *x, int incx);
template void Cublas::scal(int n, double alpha, double *x, int incx);

// complex valued instantiaions
template void Cublas::axpy(int n, muComplex alpha,
                           const muComplex *x, int incx,
                           muComplex *y, int incy);
template void Cublas::axpy(int n, muDoubleComplex alpha,
                           const muDoubleComplex *x, int incx,
                           muDoubleComplex *y, int incy);

template void Cublas::copy(int n, const muComplex *x, int incx, muComplex *y, int incy);
template void Cublas::copy(int n, const muDoubleComplex *x, int incx, muDoubleComplex *y, int incy);

template void Cublas::dot(int n, const muComplex *x, int incx,
                          const muComplex *y, int incy,
                          muComplex *result);
template void Cublas::dot(int n, const muDoubleComplex *x, int incx,
                          const muDoubleComplex *y, int incy,
                          muDoubleComplex *result);
template void Cublas::dotc(int n, const muComplex *x, int incx,
                           const muComplex *y, int incy,
                           muComplex *result);
template void Cublas::dotc(int n, const muDoubleComplex *x, int incx,
                           const muDoubleComplex *y, int incy,
                           muDoubleComplex *result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const muComplex *alpha, const muComplex *A, int lda,
                           const muComplex *x, int incx,
                           const muComplex *beta, muComplex *y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const muDoubleComplex *alpha, const muDoubleComplex *A, int lda,
                           const muDoubleComplex *x, int incx,
                           const muDoubleComplex *beta, muDoubleComplex *y, int incy);

template void Cublas::ger(int m, int n, const muComplex *alpha,
                          const muComplex *x, int incx,
                          const muComplex *y, int incy,
                          muComplex *A, int lda);
template void Cublas::ger(int m, int n, const muDoubleComplex *alpha,
                          const muDoubleComplex *x, int incx,
                          const muDoubleComplex *y, int incy,
                          muDoubleComplex *A, int lda);
template void Cublas::gerc(int m, int n, const muComplex *alpha,
                           const muComplex *x, int incx,
                           const muComplex *y, int incy,
                           muComplex *A, int lda);
template void Cublas::gerc(int m, int n, const muDoubleComplex *alpha,
                           const muDoubleComplex *x, int incx,
                           const muDoubleComplex *y, int incy,
                           muDoubleComplex *A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const muComplex *alpha, const muComplex *A, const int lda,
                               const muComplex *x, const int incx,
                               const muComplex *beta, muComplex *y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const muDoubleComplex *alpha, const muDoubleComplex *A, const int lda,
                               const muDoubleComplex *x, const int incx,
                               const muDoubleComplex *beta, muDoubleComplex *y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( mublasFillMode_t uplo, mublasOperation_t trans, mublasDiagType_t diag, int n,
                               const muComplex *A, int lda, muComplex *x, int incx, int offseta);
template void Cublas::trsv_v2( mublasFillMode_t uplo, mublasOperation_t trans, mublasDiagType_t diag, int n,
                               const muDoubleComplex *A, int lda, muDoubleComplex *x, int incx, int offseta);

template double Cublas::nrm2(int n, const muDoubleComplex *x, int incx);
template float Cublas::nrm2(int n, const muComplex *x, int incx);

template void Cublas::scal(int n, muComplex alpha, muComplex *x, int incx);
template void Cublas::scal(int n, muDoubleComplex alpha, muDoubleComplex *x, int incx);
template void Cublas::scal(int n, float alpha, muComplex *x, int incx);
template void Cublas::scal(int n, double alpha, muDoubleComplex *x, int incx);

} // namespace amgx

