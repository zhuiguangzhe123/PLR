#include <torch/extension.h>
#include <vector>

__global__ void rle_encode_kernel(
    const int64_t* x,        // 输入数据 [B, N]
    int64_t* out,            // 输出编码 [B, N]
    bool* mask,              // 有效位掩码 [B, N]
    int B, int N, int max_segments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const int64_t* seq = x + i * N;
    int64_t* seq_out = out + i * N;
    bool* seq_mask = mask + i * N;

    int idx = 0;
    int segs = 0;
    int64_t prev = seq[0];
    // printf("prev: %ld\n", prev);
    int len = 1;

    for (int j = 1; j < N; ++j) {
        if (seq[j] != prev) {
            if (segs < max_segments) {
                seq_out[idx++] = prev;
                seq_out[idx++] = len;
                segs++;
            }
            else {
                break;
            }
            prev = seq[j];
            len = 1;
        } else {
            len++;
        }
    }

    // // 最后一段
    // if (segs < max_segments) {
    //     seq_out[idx++] = prev;
    //     seq_out[idx++] = len;
    //     segs++;
    // }

    // 补原始序列
    int offset = 0;
    for (int s = 0; s < segs; ++s) {
        offset += seq_out[s * 2 + 1];
    }

    for (int j = offset; j < N && idx < N; ++j, ++idx) {
        // printf("idx: %d\n", idx);
        // printf("j: %d\n", j);
        seq_out[idx] = seq[j];
    }

    // 生成mask
    for (int j = 0; j < idx; ++j) {
        seq_mask[j] = true;  // 修正：去掉多余的偏移量计算
    }
    // 确保剩余部分为false
    for (int j = idx; j < N; ++j) {
        seq_mask[j] = false;
    }
}


// 声明 CUDA 实现
void rle_encode_cuda(torch::Tensor x, torch::Tensor out, torch::Tensor mask, int max_segments) {
    const auto B = x.size(0);
    const auto threads = 128;
    const auto blocks = (B + threads - 1) / threads;

    rle_encode_kernel<<<blocks, threads>>>(
        x.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        B,
        x.size(1),
        max_segments
    );
}

// Python 接口
std::vector<torch::Tensor> rle_encode(torch::Tensor x, int max_segments) {
    auto B = x.size(0);
    auto N = x.size(1);
    auto out = torch::zeros_like(x) - 1200;
    auto mask = torch::zeros_like(x, torch::TensorOptions().dtype(torch::kBool));
    rle_encode_cuda(x, out, mask, max_segments);
    return {out, mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rle_encode", &rle_encode, "Run-Length Encode (CUDA)");
}