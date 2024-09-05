#include "cuda_dlib_ext.cuh"

namespace dlib
{
    namespace cuda
    {
   // ----------------------------------------------------------------------------------------

        __global__ void _cuda_rms_normalize(
            float* dest,
            float* scale,
            const float* src,
            const float* gamma,
            float eps,
            size_t ns,
            size_t ks,
            size_t num
        )
        {
            for (auto nk : grid_stride_range_y(0, ns * ks))
            {
                const auto n = nk / ks;
                const auto k = nk % ks;
                const auto p_src = src + (n * ks + k) * num;
                const auto p_dest = dest + (n * ks + k) * num;
                
                float sum_squares = 0.0f;
                for (auto i : grid_stride_range(0, num))
                {
                    sum_squares += p_src[i] * p_src[i];
                }
                warp_reduce_sum(scale[n * ks + k], 1.0f / sqrtf(sum_squares / num + eps));
                __syncthreads();

                const float s = scale[n * ks + k];
                const float g = gamma[k];
                for (auto i : grid_stride_range(0, num))
                {
                    p_dest[i] = p_src[i] * s * g;
                }
            }
        }

        void rms_normalize(
            const float eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        )
        {
            const long num = src.nr() * src.nc();
            DLIB_CASSERT(
                src.k() == gamma.k() &&
                eps > 0,
                "\ngamma.k():  " << gamma.k() <<
                "\nsrc.k():    " << src.k() <<
                "\neps:  " << eps
            );

            dest.copy_size(src);
            scale.set_size(src.num_samples(), src.k());
            
            launch_kernel(_cuda_rms_normalize, max_jobs(src.num_samples() * src.k()), 
                        dest.device(), scale.device(), src.device(), gamma.device(), eps, 
                        src.num_samples(), src.k(), num);
        }

   // ----------------------------------------------------------------------------------------

        __global__ void _cuda_rms_normalize_gradient(float* out, float* gg, const float* s, const float* gi, const float* scale, const float* g, float* dscale, float eps, size_t ns, size_t num)
        {
            for (auto n : grid_stride_range_y(0, ns))
            {
                float temp_dscale = 0;
                for (auto i : grid_stride_range(0, num))
                {
                    auto idx = n * num + i;
                    const float x_hat = s[idx] * scale[n];
                    gg[i] += gi[idx] * x_hat;

                    const float dx = gi[idx] * g[i];
                    temp_dscale += dx * s[idx] * -0.5 * scale[n] * scale[n] * scale[n];
                }
                warp_reduce_atomic_add(dscale[n], temp_dscale);
            }
            __syncthreads();

            for (auto n : grid_stride_range_y(0, ns))
            {
                for (auto i : grid_stride_range(0, num))
                {
                    auto idx = n * num + i;
                    const float dx = gi[idx] * g[i];
                    out[idx] += dx * scale[n] + dscale[n] * 2 * s[idx] / num;
                }
            }
        }

        void rms_normalize_gradient(
            const double eps,
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            tensor& dscale
        )
        {
            const long num = src.k() * src.nr() * src.nc();
            DLIB_CASSERT(src.num_samples() == scale.size());
            DLIB_CASSERT(src.k() == gamma.k());
            DLIB_CASSERT(src.nr() == gamma.nr());
            DLIB_CASSERT(src.nc() == gamma.nc());
            DLIB_CASSERT(have_same_dimensions(gradient_input, src));
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad));
            DLIB_CASSERT(have_same_dimensions(gamma_grad, gamma));
            DLIB_CASSERT(eps > 0);

            gamma_grad = 0;
            dscale = 0;
            launch_kernel(_cuda_rms_normalize_gradient, max_jobs(num, src.num_samples()),
                src_grad.device(), gamma_grad.device(), src.device(),
                gradient_input.device(), scale.device(), gamma.device(),
                dscale.device(), eps, src.num_samples(), num);
        }
    }
}