/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <amgx_cusolverDn.h>

namespace amgx
{

//
// LU factorization
//
mublasStatus_t musolverDnXgetrf_bufferSize (mublasHandle_t handle,
        int m,
        int n,
        float *A,
        int lda,
        int *Lwork )
{
    return musolverSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

mublasStatus_t musolverDnXgetrf_bufferSize (mublasHandle_t handle,
        int m,
        int n,
        double *A,
        int lda,
        int *Lwork )
{
    return musolverDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

mublasStatus_t musolverDnXgetrf_bufferSize (mublasHandle_t handle,
        int m,
        int n,
        muComplex *A,
        int lda,
        int *Lwork )
{
    return musolverCgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

mublasStatus_t musolverDnXgetrf_bufferSize (mublasHandle_t handle,
        int m,
        int n,
        muDoubleComplex *A,
        int lda,
        int *Lwork )
{
    return musolverZgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

mublasStatus_t musolverDnXgetrf (mublasHandle_t handle,
                                   int m,
                                   int n,
                                   float *A,
                                   int lda,
                                   float *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return musolverSgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

mublasStatus_t musolverDnXgetrf (mublasHandle_t handle,
                                   int m,
                                   int n,
                                   double *A,
                                   int lda,
                                   double *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return musolverDgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

mublasStatus_t musolverDnXgetrf (mublasHandle_t handle,
                                   int m,
                                   int n,
                                   muComplex *A,
                                   int lda,
                                   muComplex *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return musolverCgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

mublasStatus_t musolverDnXgetrf (mublasHandle_t handle,
                                   int m,
                                   int n,
                                   muDoubleComplex *A,
                                   int lda,
                                   muDoubleComplex *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return musolverZgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

//
// solve
//
mublasStatus_t musolverDnXgetrs(mublasHandle_t handle,
                                  mublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const float *A,
                                  int lda,
                                  const int *devIpiv,
                                  float *B,
                                  int ldb,
                                  int *devInfo )
{
    return musolverSgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

mublasStatus_t musolverDnXgetrs(mublasHandle_t handle,
                                  mublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const double *A,
                                  int lda,
                                  const int *devIpiv,
                                  double *B,
                                  int ldb,
                                  int *devInfo )
{
    return musolverDgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

mublasStatus_t musolverDnXgetrs(mublasHandle_t handle,
                                  mublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const muComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  muComplex *B,
                                  int ldb,
                                  int *devInfo )
{
    return musolverCgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

mublasStatus_t musolverDnXgetrs(mublasHandle_t handle,
                                  mublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const muDoubleComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  muDoubleComplex *B,
                                  int ldb,
                                  int *devInfo )
{
    return musolverZgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

} // namespace amgx