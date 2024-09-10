#include "cuda_dlib_ext.cuh"

namespace dlib
{
    namespace cuda
    {
        __global__ void _cuda_apply_positional_encoding_optimized(
            const float* pe_data,
            float* output_data,
            const size_t te,
            const size_t nk,
            const size_t nr,
            const size_t nc
        )
        {
            for (auto i : grid_stride_range(0, te)) {
                const size_t s = i / (nk * nr * nc);
                const size_t k = (i / (nr * nc)) % nk;
                const size_t r = (i / nc) % nr;
                const size_t c = i % nc;

                const size_t offset_output = s * nk * nr * nc + k * nr * nc + r * nc + c;
                output_data[offset_output] = pe_data[r * nc + c];
            }
        }

        void apply_positional_encoding(
            const tensor& pe,
            const tensor& input,
            tensor& output
        )
        {
            DLIB_CASSERT(
                pe.num_samples() == input.nr() &&
                pe.k() == input.nc() &&
                pe.nr() == 1 &&
                pe.nc() == 1 &&
                have_same_dimensions(input, output),
                "\npe.num_samples():    " << pe.num_samples() <<
                "\npe.k():  " << pe.k() <<
                "\npe.nr(): " << pe.nr() <<
                "\npe.nc(): " << pe.nc() <<
                "\ninput.nr(): " << input.nr() <<
                "\ninput.nc(): " << input.nc()
            );
            const size_t ns = input.num_samples();
            const size_t nk = input.k();
            const size_t nr = input.nr();
            const size_t nc = input.nc();
            const size_t total_elements = ns * nk * nr * nc;

            launch_kernel(_cuda_apply_positional_encoding_optimized, max_jobs(total_elements),
                pe.device(), output.device(), total_elements, nk, nr, nc);
        }

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
            for (auto n : grid_stride_range_y(0, ns))
            {
                const auto ps = src + n * ks * num;
                float sum_squares = 0.0f;
                for (auto i : grid_stride_range(0, ks* num))
                {
                    sum_squares += ps[i] * ps[i];
                }
                warp_reduce_atomic_add(scale[n], sum_squares / (ks * num));
            }
            __syncthreads();

            for (auto n : grid_stride_range_y(0, ns))
            {
                for (auto i : grid_stride_range(0, 1))
                {
                    scale[n] = 1.0f / std::sqrt(scale[n] + eps);
                }
            }
            __syncthreads();

            for (auto n : grid_stride_range_y(0, ns))
            {
                const auto ps = src + n * ks * num;
                const auto pd = dest + n * ks * num;
                for (auto i : grid_stride_range(0, ks* num))
                {
                    pd[i] = ps[i] * scale[n] * gamma[i / num];
                }
            }
        }

        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        )
        {
            DLIB_CASSERT(
                gamma.k() == src.k() &&
                gamma.nr() == 1 &&
                gamma.nc() == 1 &&
                eps > 0,
                "\nsrc.k():    " << src.k() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\neps:  " << eps
            );

            const long ns = src.num_samples();
            const long ks = src.k();
            const long num = src.nr() * src.nc();

            dest.copy_size(src);
            scale.set_size(ns);
            scale = 0;

            launch_kernel(_cuda_rms_normalize, max_jobs(ks * num, ns),
                dest.device(), scale.device(), src.device(), gamma.device(), eps, ns, ks, num);
        }

        // ----------------------------------------------------------------------------------------

        __global__ void _cuda_rms_normalize_gradient(
            float* src_grad,
            float* gamma_grad,
            float* dscale,
            const float* src,
            const float* gradient_input,
            const float* scale,
            const float* gamma,
            size_t ns, 
            size_t ks,  
            size_t num 
        )
        {
            for (auto nk : grid_stride_range_y(0, ns * ks))
            {
                const auto n = nk / ks;
                const auto k = nk % ks;
                const auto ps = src + (n * ks + k) * num;
                const auto pgi = gradient_input + (n * ks + k) * num;
                const float scale_pow = -0.5f * std::pow(scale[n], 3.0f);
                float temp_gg = 0.0f;
                float temp_ds = 0.0f;
                for (auto i : grid_stride_range(0, num))
                {
                    const float x_hat = ps[i] * scale[n];
                    const float dx = pgi[i] * gamma[i / num];
                    temp_gg += pgi[i] * x_hat;
                    temp_ds += dx * ps[i] * scale_pow;
                }
                warp_reduce_atomic_add(gamma_grad[k], temp_gg);
                warp_reduce_atomic_add(dscale[n], temp_ds);
            }
            __syncthreads();

            const float invnum = 1.0f / (ks * num);
            for (auto n : grid_stride_range_y(0, ns))
            {
                const auto ps = src + n * ks * num;
                const auto pgi = gradient_input + n * ks * num;
                const auto psg = src_grad + n * ks * num;
                for (auto i : grid_stride_range(0, ks * num))
                {
                    const float dx = pgi[i] * gamma[i / num];
                    psg[i] += dx * scale[n] + dscale[n] * 2 * ps[i] * invnum;
                }
            }
        }

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
        )
        {            
            DLIB_CASSERT(src.num_samples() == scale.size());
            DLIB_CASSERT(have_same_dimensions(gamma, gamma_grad));
            DLIB_CASSERT(gamma.k() == src.k());
            DLIB_CASSERT(gamma.nr() == 1);
            DLIB_CASSERT(gamma.nc() == 1);
            DLIB_CASSERT(have_same_dimensions(gradient_input, src));
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad));

            const long ns = src.num_samples();
            const long ks = src.k();
            const long num = src.nr() * src.nc();

            gamma_grad = 0;
            dscale.copy_size(scale);
            dscale = 0;

            launch_kernel(_cuda_rms_normalize_gradient, max_jobs(ks * num, ns),
                src_grad.device(), gamma_grad.device(), dscale.device(),
                src.device(), gradient_input.device(), scale.device(), gamma.device(),
                ns, ks, num);
        }

        __global__ void transpose_kernel(
            float* dest,
            const float* src,
            const long num_samples,
            const long k,
            const long src_nr,
            const long src_nc
        )
        {
            const long n = blockIdx.x * blockDim.x + threadIdx.x;
            const long k_idx = blockIdx.y * blockDim.y + threadIdx.y;
            const long r = blockIdx.z * blockDim.z + threadIdx.z;

            if (n < num_samples && k_idx < k && r < src_nr)
            {
                for (long c = 0; c < src_nc; ++c)
                {
                    dest[((n * k + k_idx) * src_nc + c) * src_nr + r] =
                        src[((n * k + k_idx) * src_nr + r) * src_nc + c];
                }
            }
        }

        __global__ void transpose_add_kernel(
            float* dest,
            const float* src,
            const long num_samples,
            const long k,
            const long src_nr,
            const long src_nc
        )
        {
            const long n = blockIdx.x * blockDim.x + threadIdx.x;
            const long k_idx = blockIdx.y * blockDim.y + threadIdx.y;
            const long r = blockIdx.z * blockDim.z + threadIdx.z;

            if (n < num_samples && k_idx < k && r < src_nr)
            {
                for (long c = 0; c < src_nc; ++c)
                {
                    atomicAdd(&dest[((n * k + k_idx) * src_nc + c) * src_nr + r],
                        src[((n * k + k_idx) * src_nr + r) * src_nc + c]);
                }
            }
        }

        void transpose(
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == src.k() &&
                dest.nr() == src.nc() &&
                dest.nc() == src.nr());

            cudaDeviceProp prop;
            int device;
            cudaGetDevice(&device);
            cudaGetDeviceProperties(&prop, device);

            // Dynamic block size adjustment
            dim3 block(32, 32, 1);
            while (block.x * block.y * block.z > (unsigned int)(prop.maxThreadsPerBlock)) {
                if (block.z > 1) block.z /= 2;
                else if (block.y > 1) block.y /= 2;
                else block.x /= 2;
            }

            const dim3 grid(
                (src.num_samples() + block.x - 1) / block.x,
                (src.k() + block.y - 1) / block.y,
                (src.nr() + block.z - 1) / block.z
            );

            transpose_kernel << <grid, block >> > (dest.device(), src.device(),
                src.num_samples(), src.k(), src.nr(), src.nc());
        }

        void transpose_add(
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == src.k() &&
                dest.nr() == src.nc() &&
                dest.nc() == src.nr());

            cudaDeviceProp prop;
            int device;
            cudaGetDevice(&device);
            cudaGetDeviceProperties(&prop, device);

            // Dynamic block size adjustment
            dim3 block(32, 32, 1);
            while (block.x * block.y * block.z > (unsigned int)(prop.maxThreadsPerBlock)) {
                if (block.z > 1) block.z /= 2;
                else if (block.y > 1) block.y /= 2;
                else block.x /= 2;
            }

            const dim3 grid(
                (src.num_samples() + block.x - 1) / block.x,
                (src.k() + block.y - 1) / block.y,
                (src.nr() + block.z - 1) / block.z
            );

            transpose_add_kernel << <grid, block >> > (dest.device(), src.device(),
                src.num_samples(), src.k(), src.nr(), src.nc());
        }

        __global__ void batch_multiply_kernel(
            float* out,
            const float* a,
            const float* b,
            bool a_trans,
            bool b_trans,
            long num_samples,
            long num_channels,
            long a_nr, long a_nc,
            long b_nr, long b_nc,
            long out_nr, long out_nc
        )
        {
            const long K = a_trans ? a_nr : a_nc;
            const long M = out_nr;
            const long N = out_nc;

            const long sample = blockIdx.x;
            const long channel = blockIdx.y;
            const long row = blockIdx.z * blockDim.y + threadIdx.y;
            const long col = threadIdx.x;

            if (row < M && col < N)
            {
                const float* a_data = a + ((sample * num_channels + channel) * a_nr * a_nc);
                const float* b_data = b + ((sample * num_channels + channel) * b_nr * b_nc);
                float* out_data = out + ((sample * num_channels + channel) * M * N);

                float sum = 0;
                for (long i = 0; i < K; ++i)
                {
                    long a_index = a_trans ? (i * a_nc + row) : (row * a_nc + i);
                    long b_index = b_trans ? (col * b_nr + i) : (i * b_nc + col);
                    sum += a_data[a_index] * b_data[b_index];
                }
                out_data[row * N + col] = sum;
            }
        }

        void batch_multiply(
            tensor& out,
            const tensor& a,
            bool a_trans,
            const tensor& b,
            bool b_trans
        )
        {
            const long num_samples = a.num_samples();
            const long num_channels = a.k();
            const long a_nr = a.nr();
            const long a_nc = a.nc();
            const long b_nr = b.nr();
            const long b_nc = b.nc();

            // Vérifications des dimensions
            DLIB_CASSERT(a.num_samples() == b.num_samples());
            DLIB_CASSERT(a.num_samples() == out.num_samples());
            DLIB_CASSERT(a.k() == b.k());
            DLIB_CASSERT(a.k() == out.k());
            DLIB_CASSERT(a_trans ? a_nr : a_nc == (b_trans ? b_nc : b_nr));
            DLIB_CASSERT(out.nr() == (a_trans ? a_nc : a_nr));
            DLIB_CASSERT(out.nc() == (b_trans ? b_nr : b_nc));

            const long M = out.nr();
            const long N = out.nc();

            cudaDeviceProp prop;
            int device;
            cudaGetDevice(&device);
            cudaGetDeviceProperties(&prop, device);

            dim3 block(32, 32, 1);
            while (block.x * block.y * block.z > (unsigned int)(prop.maxThreadsPerBlock)) {
                if (block.z > 1) block.z /= 2;
                else if (block.y > 1) block.y /= 2;
                else block.x /= 2;
            }
            dim3 grid(num_samples, num_channels, (M + block.y - 1) / block.y);

            batch_multiply_kernel << <grid, block >> > (out.device(), a.device(), b.device(),
                a_trans, b_trans, num_samples, num_channels, a_nr, a_nc, b_nr, b_nc, M, N);
        }
    }
}