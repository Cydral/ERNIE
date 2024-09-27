#ifndef DLIB_DNN_CuDA_EXT_H_
#define DLIB_DNN_CuDA_EXT_H_

#include <dlib/cuda/cuda_utils.h>
#include <dlib/cuda/cuda_dlib.h>
#include <dlib/cuda/cudnn_dlibapi.h>
#include <dlib/cuda/cublas_dlibapi.h>
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
            bool add_to,
            tensor& dest,
            const tensor& src
        );

        void split_columns(
            bool add_to,
            tensor& dest,
            const tensor& src,
            const long num_heads
        );

        void merge_columns(
            bool add_to,
            tensor& dest,
            const tensor& src
        );

        void reorg2(
            bool add_to,
            tensor& dest,
            const int row_stride,
            const int col_stride,
            const tensor& src
        );

        void reorg_gradient2(
            bool add_to,
            tensor& grad,
            const int row_stride,
            const int col_stride,
            const tensor& gradient_input
        );

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& embs
        );

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& grads,
            const tensor& freqs,
            float learning_rate,
            bool scale
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
