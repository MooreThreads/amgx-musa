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
#pragma once

#include <basic_types.h>

// add operator
__host__ __device__ __inline__ muComplex operator+(const muComplex &lhs, const muComplex &rhs)
{
    return muCaddf(lhs, rhs);
}
__host__ __device__ __inline__ muComplex operator+(const muComplex &lhs, const muDoubleComplex &rhs)
{
    return make_muComplex(muCrealf(lhs) + muCreal(rhs), muCimagf(lhs) + muCimag(rhs));
}
__host__ __device__ __inline__ muComplex operator+(const muComplex &lhs, const float &rhs)
{
    return make_muComplex(muCrealf(lhs) + rhs, muCimagf(lhs));
}
__host__ __device__ __inline__ muComplex operator+(const muComplex &lhs, const double &rhs)
{
    return make_muComplex(muCrealf(lhs) + rhs, muCimagf(lhs));
}

__host__ __device__ __inline__ muDoubleComplex operator+(const muDoubleComplex &lhs, const muComplex &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) + muCrealf(rhs), muCimag(lhs) + muCimagf(rhs));
}
__host__ __device__ __inline__ muDoubleComplex operator+(const muDoubleComplex &lhs, const muDoubleComplex &rhs)
{
    return muCadd(lhs, rhs);
}
__host__ __device__ __inline__ muDoubleComplex operator+(const muDoubleComplex &lhs, const float &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) + rhs, muCimag(lhs));
}
__host__ __device__ __inline__ muDoubleComplex operator+(const muDoubleComplex &lhs, const double &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) + rhs, muCimag(lhs));
}

// sub operator
__host__ __device__ __inline__ muComplex operator-(const muComplex &lhs, const muComplex &rhs)
{
    return muCsubf(lhs, rhs);
}
__host__ __device__ __inline__ muComplex operator-(const muComplex &lhs, const muDoubleComplex &rhs)
{
    return make_muComplex(muCrealf(lhs) - muCreal(rhs), muCimagf(lhs) - muCimag(rhs));
}
__host__ __device__ __inline__ muComplex operator-(const muComplex &lhs, const float &rhs)
{
    return make_muComplex(muCrealf(lhs) - rhs, muCimagf(lhs));
}
__host__ __device__ __inline__ muComplex operator-(const muComplex &lhs, const double &rhs)
{
    return make_muComplex(muCrealf(lhs) - rhs, muCimagf(lhs));
}

__host__ __device__ __inline__ muDoubleComplex operator-(const muDoubleComplex &lhs, const muComplex &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) - muCrealf(rhs), muCimag(lhs) - muCimagf(rhs));
}
__host__ __device__ __inline__ muDoubleComplex operator-(const muDoubleComplex &lhs, const muDoubleComplex &rhs)
{
    return muCsub(lhs, rhs);
}
__host__ __device__ __inline__ muDoubleComplex operator-(const muDoubleComplex &lhs, const float &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) - rhs, muCimag(lhs));
}
__host__ __device__ __inline__ muDoubleComplex operator-(const muDoubleComplex &lhs, const double &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) - rhs, muCimag(lhs));
}


// multiply operator
__host__ __device__ __inline__ muComplex operator*(const muComplex &lhs, const muComplex &rhs)
{
    return muCmulf(lhs, rhs);
}
__host__ __device__ __inline__ muComplex operator*(const muComplex &lhs, const muDoubleComplex &rhs)
{
    // basically copy of muComplex.h muCmulf
    muComplex prod;
    prod = make_muComplex   ((muCrealf(lhs) * muCreal(rhs)) -
                             (muCimagf(lhs) * muCimag(rhs)),
                             (muCrealf(lhs) * muCimag(rhs)) +
                             (muCimagf(lhs) * muCreal(rhs)));
    return prod;
}
__host__ __device__ __inline__ muComplex operator*(const muComplex &lhs, const float &rhs)
{
    return make_muComplex(muCrealf(lhs) * rhs, muCimagf(lhs) * rhs);
}
__host__ __device__ __inline__ muComplex operator*(const muComplex &lhs, const double &rhs)
{
    return make_muComplex(muCrealf(lhs) * rhs, muCimagf(lhs) * rhs);
}


__host__ __device__ __inline__ muDoubleComplex operator*(const muDoubleComplex &lhs, const muComplex &rhs)
{
    muDoubleComplex prod;
    prod = make_muDoubleComplex ((muCreal(lhs) * muCrealf(rhs)) -
                                 (muCimag(lhs) * muCimagf(rhs)),
                                 (muCreal(lhs) * muCimagf(rhs)) +
                                 (muCimag(lhs) * muCrealf(rhs)));
    return prod;
}
__host__ __device__ __inline__ muDoubleComplex operator*(const muDoubleComplex &lhs, const muDoubleComplex &rhs)
{
    return muCmul(lhs, rhs);
}
__host__ __device__ __inline__ muDoubleComplex operator*(const muDoubleComplex &lhs, const float &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) * rhs, muCimag(lhs));
}
__host__ __device__ __inline__ muDoubleComplex operator*(const muDoubleComplex &lhs, const double &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) * rhs, muCimag(lhs));
}

// div operator
__host__ __device__ __inline__ muComplex operator/(const muComplex &lhs, const muComplex &rhs)
{
    return muCdivf(lhs, rhs);
}
__host__ __device__ __inline__ muComplex operator/(const muComplex &lhs, const muDoubleComplex &rhs)
{
    return muCdivf(lhs, make_muComplex(muCreal(rhs), muCimag(rhs)));
}
__host__ __device__ __inline__ muComplex operator/(const muComplex &lhs, const float &rhs)
{
    return make_muComplex(muCrealf(lhs) / rhs, muCimagf(lhs) / rhs);
}
__host__ __device__ __inline__ muComplex operator/(const muComplex &lhs, const double &rhs)
{
    return make_muComplex(muCrealf(lhs) / rhs, muCimagf(lhs) / rhs);
}

__host__ __device__ __inline__ muDoubleComplex operator/(const muDoubleComplex &lhs, const muComplex &rhs)
{
    return muCdiv(lhs, make_muDoubleComplex(muCrealf(rhs), muCimagf(rhs)));
}
__host__ __device__ __inline__ muDoubleComplex operator/(const muDoubleComplex &lhs, const muDoubleComplex &rhs)
{
    return muCdiv(lhs, rhs);
}
__host__ __device__ __inline__ muDoubleComplex operator/(const muDoubleComplex &lhs, const float &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) / rhs, muCimag(lhs) / rhs);
}
__host__ __device__ __inline__ muDoubleComplex operator/(const muDoubleComplex &lhs, const double &rhs)
{
    return make_muDoubleComplex(muCreal(lhs) / rhs, muCimag(lhs) / rhs);
}
