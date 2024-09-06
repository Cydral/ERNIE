#ifndef DLIB_DNN_CuDA_EXT_H_
#define DLIB_DNN_CuDA_EXT_H_

#include <dlib/cuda/cuda_utils.h>
#include <dlib/cuda/cuda_dlib.h>
#include <dlib/cuda/cudnn_dlibapi.h>
#include <math_constants.h>

namespace dlib
{
    namespace cuda
    {
#ifdef DLIB_USE_CUDA
        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        );

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
        );

        void transpose(
            tensor& dest,
            const tensor& src
        );

        void transpose_add(
            tensor& dest,
            const tensor& src
        );

        void batch_multiply(
            tensor& out,
            const tensor& a,
            bool a_trans,
            const tensor& b,
            bool b_trans
        );
#else // if DLIB_USE_CUDA NOT DEFINED
#endif // DLIB_USE_CUDA
    }
}

#endif // DLIB_DNN_CuDA_EXT_H_
