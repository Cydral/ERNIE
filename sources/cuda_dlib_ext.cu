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

        __global__ void _cuda_transpose(size_t dsize, size_t dk, size_t dnr, size_t dnc, float* d,
            size_t sk, size_t snr, int snc, const float* s, const bool add_to)
        {
            const auto plane_size = dnr * dnc;
            const auto sample_size = dk * plane_size;
            for (auto i : grid_stride_range(0, dsize))
            {
                const auto n = i / sample_size;
                const auto idx = i % plane_size;
                const auto in_k = (i / plane_size) % dk;
                const auto in_r = idx % dnc;
                const auto in_c = idx / dnc;

                const auto in_idx = ((n * sk + in_k) * snr + in_r) * snc + in_c;
                if (add_to) d[i] += s[in_idx];
                else d[i] = s[in_idx];
            }
        }

        void transpose(
            bool add_to,
            tensor& dest,
            const tensor& src            
        )
        {
            DLIB_CASSERT(is_same_object(dest, src) == false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == src.k() &&
                dest.nr() == src.nc() &&
                dest.nc() == src.nr(),
                "Incompatible tensor dimensions.");

            launch_kernel(_cuda_transpose, max_jobs(dest.size()), dest.size(),
                dest.k(), dest.nr(), dest.nc(), dest.device(),
                src.k(), src.nr(), src.nc(), src.device(), add_to);
        }
           
        __global__ void cuda_split_columns(
            size_t size,
            size_t snr,
            size_t snc,
            size_t nh,
            size_t hd,
            const float* input,
            float* output,
            const bool add_to
        )
        {
            for (auto i : grid_stride_range(0, size))
            {
                const auto n = i / (nh * snr * hd);
                const auto r = i % (nh * snr * hd);
                const auto h = r / (snr * hd);
                const auto s = (r / hd) % snr;
                const auto d = r % hd;

                const auto input_idx = ((n * snr + s) * snc) + (h * hd + d);
                if (add_to) output[i] += input[input_idx];
                else output[i] = input[input_idx];

            }
        }

        void split_columns(
            bool add_to,
            tensor& dest,
            const tensor& src,
            const long num_heads)
        {
            DLIB_CASSERT(is_same_object(dest, src) == false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == num_heads &&
                src.k() == 1 &&
                dest.nc() == (src.nc() / num_heads) &&
                src.nc() % num_heads == 0,
                "Incompatible tensor dimensions.");

            launch_kernel(cuda_split_columns, max_jobs(dest.size()),
                dest.size(), src.nr(), src.nc(), num_heads, src.nc() / num_heads,
                src.device(), dest.device(), add_to);
        }

        __global__ void cuda_merge_columns(
            size_t size,
            size_t dnr,
            size_t dnc,
            size_t nh,
            size_t hd,
            const float* input,
            float* output,
            const bool add_to
        )
        {
            for (auto i : grid_stride_range(0, size))
            {
                const auto n = i / (dnr * dnc);
                const auto r = (i / dnc) % dnr;
                const auto c = i % dnc;
                const auto h = c / hd;
                const auto d = c % hd;

                const auto input_idx = ((n * nh + h) * dnr + r) * hd + d;
                if (add_to) output[i] += input[input_idx];
                else output[i] = input[input_idx];
            }
        }

        void merge_columns(
            bool add_to,
            tensor& dest,
            const tensor& src)
        {
            DLIB_CASSERT(is_same_object(dest, src) == false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == 1 &&
                src.k() > 1 &&
                dest.nr() == src.nr() &&
                dest.nc() == (src.nc() * src.k()),
                "Incompatible tensor dimensions.");

            launch_kernel(cuda_merge_columns, max_jobs(dest.size()),
                dest.size(), dest.nr(), dest.nc(), src.k(), src.nc(),
                src.device(), dest.device(), add_to);
        }

        __global__ void _cuda_reorg2(size_t dsize, size_t dk, size_t dnr, size_t dnc, float* d,
            size_t sk, size_t snr, int snc, const float* s,
            const size_t row_stride, const size_t col_stride, const bool add_to)
        {
            const auto out_plane_size = dnr * dnc;
            const auto out_sample_size = dk * out_plane_size;
            for (auto i : grid_stride_range(0, dsize))
            {
                const auto n = i / out_sample_size;
                const auto out_idx = i % out_sample_size;
                const auto out_k = out_idx / out_plane_size;
                const auto out_rc = out_idx % out_plane_size;
                const auto out_r = out_rc / dnc;
                const auto out_c = out_rc % dnc;

                const auto in_k = out_k % sk;
                const auto in_r = out_r * row_stride + (out_k / sk) / col_stride;
                const auto in_c = out_c * col_stride + (out_k / sk) % col_stride;

                const auto in_idx = ((n * sk + in_k) * snr + in_r) * snc + in_c;
                if (add_to) d[i] += s[in_idx];
                else d[i] = s[in_idx];
            }
        }

        void reorg2(
            bool add_to,
            tensor& dest,
            const int row_stride,
            const int col_stride,
            const tensor& src
        )
        {
            DLIB_CASSERT(!is_same_object(dest, src), "Destination and source must be distinct objects.");
            DLIB_CASSERT(src.nr() % row_stride == 0, "The number of rows in src must be divisible by row_stride.");
            DLIB_CASSERT(src.nc() % col_stride == 0, "The number of columns in src must be divisible by col_stride.");
            DLIB_CASSERT(dest.num_samples() == src.num_samples(), "The number of samples must match.");
            DLIB_CASSERT(dest.k() == src.k() * row_stride * col_stride, "The number of channels must match.");
            DLIB_CASSERT(dest.nr() == src.nr() / row_stride, "The number of rows must match.");
            DLIB_CASSERT(dest.nc() == src.nc() / col_stride, "The number of columns must match.");

            launch_kernel(_cuda_reorg2, dest.size(), dest.k(), dest.nr(), dest.nc(), dest.device(),
                src.k(), src.nr(), src.nc(), src.device(), row_stride, col_stride, add_to);
        }

        __global__ void _cuda_reorg_gradient2(size_t ssize, size_t dk, size_t dnr, size_t dnc, float* d,
            size_t sk, size_t snr, int snc, const float* s, const size_t row_stride, const size_t col_stride,
            const bool add_to
        )
        {
            for(auto i : grid_stride_range(0, ssize))
            {
                const auto n = i / (sk * snr * snc);
                const auto sample_idx = i % (sk * snr * snc);
                const auto in_k = (sample_idx / (snr * snc)) % sk;
                const auto in_r = (sample_idx / snc) % snr;
                const auto in_c = sample_idx % snc;

                const auto out_k = in_k % dk;
                const auto out_r = in_r * row_stride + (in_k / dk) / col_stride;
                const auto out_c = in_c * col_stride + (in_k / dk) % col_stride;
                const auto out_idx = ((n * dk + out_k) * dnr + out_r) * dnc + out_c;

                if (add_to) d[out_idx] += s[i];
                else d[out_idx] = s[i];
            }
        }

        void reorg_gradient2(
            bool add_to,
            tensor& grad,
            const int row_stride,
            const int col_stride,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(!is_same_object(grad, gradient_input), "Grad and gradient_input must be distinct objects.");
            DLIB_CASSERT(grad.nr() % row_stride == 0, "The number of rows in grad must be divisible by row_stride.");
            DLIB_CASSERT(grad.nc() % col_stride == 0, "The number of columns in grad must be divisible by col_stride.");
            DLIB_CASSERT(grad.num_samples() == gradient_input.num_samples(), "The number of samples in grad and gradient_input must match.");
            DLIB_CASSERT(grad.k() == gradient_input.k() / row_stride / col_stride, "The number of channels in grad must be gradient_input.k() divided by row_stride and col_stride.");
            DLIB_CASSERT(grad.nr() == gradient_input.nr() * row_stride, "The number of rows in grad must be gradient_input.nr() multiplied by row_stride.");
            DLIB_CASSERT(grad.nc() == gradient_input.nc() * col_stride, "The number of columns in grad must be gradient_input.nc() multiplied by col_stride.");

            launch_kernel(_cuda_reorg_gradient2, gradient_input.size(), grad.k(), grad.nr(), grad.nc(), grad.device(),
                gradient_input.k(), gradient_input.nr(), gradient_input.nc(), gradient_input.device(),
                row_stride, col_stride, add_to);
        }

        __global__ void cuda_embeddings(
            size_t ssize, size_t ns, size_t nk, size_t nr, size_t nc,
            float* d, const float* s, const float* e, size_t es
        )
        {
            for (auto i : grid_stride_range(0, ssize))
            {
                const size_t nk_nr = nk * nr;
                const auto s_idx = i / nk_nr;
                const auto k = (i % nk_nr) / nr;
                const auto r = i % nr;

                const unsigned long t_idx = static_cast<unsigned long>(s[s_idx * nk_nr + k * nr + r]);
                const size_t base_idx = s_idx * nk_nr * nc + k * nr * nc + r * nc;

                if (t_idx < es)
                {
                    const size_t emb_base_idx = t_idx * nc;
                    for (long c = 0; c < nc; ++c) d[base_idx + c] = e[emb_base_idx + c];
                }
                else
                {
                    for (long c = 0; c < nc; ++c) d[base_idx + c] = 1;
                }
            }
        }

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& emb
        )
        {
            DLIB_CASSERT(
                src.nr() > 0 &&
                emb.num_samples() > 0 &&
                emb.k() > 0 &&
                emb.nr() == 1 &&
                emb.nc() == 1,
                "\nsrc.num_samples(): " << src.num_samples() <<
                "\nsrc.k(): " << src.k() <<
                "\nsrc.nr(): " << src.nr() <<
                "\nsrc.nc(): " << src.nc() <<
                "\nemb.num_samples(): " << emb.num_samples() <<
                "\nemb.k(): " << emb.k() <<
                "\nemb.nr(): " << emb.nr() <<
                "\nemb.nc(): " << emb.nc()
            );

            const long ns = dest.num_samples();
            const long nk = dest.k();
            const long nr = dest.nr();
            const long nc = dest.nc();

            launch_kernel(_cuda_embeddings, ns * nk * nr, ns, nk, nr, nc,
                dest.device(), src.device(), emb.device(), emb.num_samples());
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