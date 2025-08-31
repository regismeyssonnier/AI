
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <functional>
#include <vector>
#include <unordered_set>
#include <set>
#include <random>
#include <string>

// Kernel simple pour mise à jour SGD sur GPU
__global__ void sgd_update_kernel(float* data, const float* grad, float lr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] -= lr * grad[idx];
    }
}

//Kernel
// --- Forward kernels ---
__global__ void add_kernel(const float* A, const float* B, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = A[idx] + B[idx];
}

__global__ void sub_kernel(const float* A, const float* B, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = A[idx] - B[idx];
}

__global__ void mul_kernel(const float* A, const float* B, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = A[idx] * B[idx];
}

__global__ void div_kernel(const float* A, const float* B, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = A[idx] / B[idx];
}

__global__ void neg_kernel(const float* A, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = -A[idx];
}

__global__ void relu_kernel(const float* A, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = fmaxf(0.0f, A[idx]);
}

__global__ void sigmoid_kernel(const float* A, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = 1.0f / (1.0f + expf(-A[idx]));
}

__global__ void tanh_kernel(const float* A, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = tanhf(A[idx]);
}

__global__ void pow_elem_kernel(const float* A, const float* B, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = powf(A[idx], B[idx]);
}

// --- Backward kernels ---
__global__ void add_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_a[idx] += grad_out[idx];
        grad_b[idx] += grad_out[idx];
    }
}

__global__ void sub_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_a[idx] += grad_out[idx];
        grad_b[idx] -= grad_out[idx];
    }
}

__global__ void mul_backward_kernel(const float* grad_out, const float* other_data, float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_self[idx] += grad_out[idx] * other_data[idx];
}

__global__ void div_backward_kernel(const float* grad_out, const float* self_data, const float* other_data, float* grad_self, float* grad_other, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_self[idx] += grad_out[idx] / other_data[idx];
        grad_other[idx] -= grad_out[idx] * self_data[idx] / (other_data[idx] * other_data[idx]);
    }
}

__global__ void neg_backward_kernel(const float* grad_out, float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_self[idx] -= grad_out[idx];
}

__global__ void relu_backward_kernel(const float* grad_out, const float* self_data, float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_self[idx] += (self_data[idx] > 0.0f ? grad_out[idx] : 0.0f);
}

__global__ void sigmoid_backward_kernel(const float* grad_out, const float* out_data, float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float s = out_data[idx];
        grad_self[idx] += grad_out[idx] * s * (1.0f - s);
    }
}

__global__ void tanh_backward_kernel(const float* grad_out, const float* out_data, float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float t = out_data[idx];
        grad_self[idx] += grad_out[idx] * (1.0f - t * t);
    }
}

__global__ void pow_elem_grad_kernel(float* grad_A, float* grad_B, const float* A, const float* B, const float* out, const float* grad_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        float go = grad_out[idx];

        grad_A[idx] += b * powf(a, b - 1.0f) * go;
        if (a > 0.0f)
            grad_B[idx] += logf(a) * out[idx] * go;
    }
}

// Forward: out[i] = base[i]^exp
__global__ void pow_kernel(const float* base, float exp, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = powf(base[idx], exp);
}

// Backward: grad_base[i] += exp * base[i]^(exp-1) * grad_out[i]
__global__ void pow_grad_kernel(float* grad_base, const float* base, float exp,
    const float* grad_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_base[idx] += exp * powf(base[idx], exp - 1.0f) * grad_out[idx];
}

__global__ void set_ones_kernel(float* grad, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad[idx] = 1.0f;
}


bool NOGRAD = false;

class UnityGPU : public std::enable_shared_from_this<UnityGPU> {
public:
    int size;
    float* data;
    float* grad;
    std::vector<std::weak_ptr<UnityGPU>> _prev;
    std::function<void()> _backward;
    std::string info;
    std::shared_ptr<UnityGPU> _pre_activation;

    UnityGPU(int n) : size(n) {
        cudaMalloc(&data, n * sizeof(float));
        cudaMalloc(&grad, n * sizeof(float));
        cudaMemset(data, 0, n * sizeof(float));
        cudaMemset(grad, 0, n * sizeof(float));
        _backward = []() {};
        info += std::to_string(n);
    }

    UnityGPU(int n, float init_val) : size(n) {
        cudaMalloc(&data, n * sizeof(float));
        cudaMalloc(&grad, n * sizeof(float));
        std::vector<float> h(n, init_val);
        cudaMemcpy(data, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(grad, 0, n * sizeof(float));
        _backward = []() {};
        info += std::to_string(n);
    }


    UnityGPU() : size(0), data(nullptr), grad(nullptr) {
        _backward = []() {};
    }


    ~UnityGPU() {
        cudaFree(data);
        cudaFree(grad);
    }

    void zero_grad() {
        cudaMemset(grad, 0, size * sizeof(float));
    }

    void backward2() {
        
        _backward(); // appelle le backward défini lors de l'opération
    }

    void backward() {
        std::vector<std::shared_ptr<UnityGPU>> topo;
        std::unordered_set<UnityGPU*> visited;

        std::function<void(std::shared_ptr<UnityGPU>)> build_topo = [&](std::shared_ptr<UnityGPU> v) {
            if (v && !visited.count(v.get())) {
                visited.insert(v.get());
                for (auto& child_weak : v->_prev) {
                    if (auto child = child_weak.lock()) build_topo(child);
                }
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        // grad = 1 sur GPU
        /*cudaMemset(this->grad, 0, size * sizeof(float));
        if (size == 1) {
            float one = 1.0f;
            cudaMemcpy(this->grad, &one, sizeof(float), cudaMemcpyHostToDevice);
         }
        else {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            set_ones_kernel << <blocks, threads >> > (grad, size);
        }*/

        // Backward propagation
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it && (*it)->_backward) {
                //std::cout << (*it)->info << std::endl;
                (*it)->_backward();

            }
        }
    }


    // ---------------- Binary Ops ----------------
    std::shared_ptr<UnityGPU> add(const std::shared_ptr<UnityGPU>& other) {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        add_kernel <<<blocks, threads >>> (data, other->data, out->data, size);

        if (!NOGRAD) {
            out->_prev = { shared_from_this(), other };
            out->_backward = [self = shared_from_this(), other, out]() {
                int threads = 256;
                int blocks = (self->size + threads - 1) / threads;
                add_backward_kernel << <blocks, threads >> > (out->grad, self->grad, other->grad, self->size);
            };
        }
        return out;
    }

    // Sub, mul, div, neg, relu, sigmoid, tanh, pow se font pareil en appelant leurs kernels forward et backward


    std::shared_ptr<UnityGPU> sub(const std::shared_ptr<UnityGPU>& other) {
        return add(other->neg());
    }

    std::shared_ptr<UnityGPU> mul(const std::shared_ptr<UnityGPU>& other) {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        // Forward
        mul_kernel << <blocks, threads >> > (data, other->data, out->data, size);

        if (!NOGRAD) {
            out->_prev = { shared_from_this(), other };
            out->_backward = [self = shared_from_this(), other, out]() {
                int threads = 256;
                int blocks = (self->size + threads - 1) / threads;

                // Gradient par rapport à A
                mul_backward_kernel << <blocks, threads >> > (out->grad, other->data, self->grad, self->size);

                // Gradient par rapport à B
                mul_backward_kernel << <blocks, threads >> > (out->grad, self->data, other->grad, other->size);

            };
        }

        return out;
    }


    // Pow avec un scalaire
    std::shared_ptr<UnityGPU> pow(float exponent) {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        pow_kernel << <blocks, threads >> > (data, exponent, out->data, size);

        if (!NOGRAD) {
            out->_prev = { shared_from_this() };
            out->_backward = [self = shared_from_this(), exponent, out]() {
                pow_grad_kernel <<<(self->size + 255) / 256, 256 >>> (self->grad, self->data, exponent, out->grad, self->size);
            };
        }
        return out;
    }

    // Pow avec un autre UnityGPU (élément par élément)
    // Pow avec un autre UnityGPU (élément par élément)
    std::shared_ptr<UnityGPU> pow(const std::shared_ptr<UnityGPU>& other) {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        // Forward
        pow_elem_kernel << <blocks, threads >> > (data, other->data, out->data, size);

        // Backward
        if (!NOGRAD) {
            out->_prev = { shared_from_this(), other };
            out->_backward = [self = shared_from_this(), other, out]() {
                int threads = 256;
                int blocks = (self->size + threads - 1) / threads;
                pow_elem_grad_kernel << <blocks, threads >> > (
                    self->grad,      // gradient par rapport à la base
                    other->grad,     // gradient par rapport à l’exposant
                    self->data,      // base A
                    other->data,     // exposant B
                    out->data,       // résultat forward
                    out->grad,       // gradient du résultat
                    self->size       // taille N
                    );
            };
        }

        return out;
    }



    std::shared_ptr<UnityGPU> div(const std::shared_ptr<UnityGPU>& other) {
        auto inv = other->pow(-1.0f);
        return mul(inv);
    }

    std::shared_ptr<UnityGPU> neg() {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        neg_kernel << <blocks, threads >> > (data, out->data, size);
        if (!NOGRAD) {
            out->_prev = { shared_from_this() };
            out->_backward = [self = shared_from_this(), out]() {
                neg_backward_kernel << <(self->size + 255) / 256, 256 >> > (self->grad, out->grad, self->size);
            };
        }
        return out;
    }

    // ---------------- Activations ----------------
    std::shared_ptr<UnityGPU> relu() {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_kernel << <blocks, threads >> > (data, out->data, size);
        if (!NOGRAD) {
            out->_prev = { shared_from_this() };
            out->_backward = [self = shared_from_this(), out]() {
                relu_backward_kernel << <(self->size + 255) / 256, 256 >> > (out->grad, self->data, self->grad, self->size);
            };
        }
        return out;
    }

    std::shared_ptr<UnityGPU> tanh() {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        tanh_kernel << <blocks, threads >> > (data, out->data, size);
        if (!NOGRAD) {
            out->_prev = { shared_from_this() };
            out->_backward = [self = shared_from_this(), out]() {
                tanh_backward_kernel << <(self->size + 255) / 256, 256 >> > (out->grad, out->data, self->grad, self->size);
            };
        }
        return out;
    }

    std::shared_ptr<UnityGPU> sigmoid() {
        auto out = std::make_shared<UnityGPU>(size);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sigmoid_kernel <<<blocks, threads >>> (data, out->data, size);
        if (!NOGRAD) {
            out->_prev = { shared_from_this() };
            out->_backward = [self = shared_from_this(), out]() {
                sigmoid_backward_kernel << <(self->size + 255) / 256, 256 >> > (out->grad, out->data, self->grad, self->size);
            };
        }
        return out;
    }

    // ---------------- Update ----------------
    void update(float lr) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sgd_update_kernel << <blocks, threads >> > (data, grad, lr, size);
    }
};


// ---------------- Module ----------------
class Module {
public:
    virtual void zero_grad() {
        for (auto& p : parameters())
            cudaMemset(p->grad, 0, sizeof(float) * p->size);
    }

    virtual std::vector<std::shared_ptr<UnityGPU>> parameters() { return {}; }
    virtual ~Module() = default;
};

// ---------------- Node ----------------
/*
class Node : public Module {
public:
    std::vector<std::shared_ptr<UnityGPU>> w;
    std::shared_ptr<UnityGPU> b;
    int nonlin; // 0=linear, 1=ReLU, 2=Sigmoid

    Node(int nin, int nonlin_ = 1) : nonlin(nonlin_) {
        for (int i = 0; i < nin; i++)
            w.push_back(std::make_shared<UnityGPU>(1, ((float)rand() / RAND_MAX) * 2 - 1));
        b = std::make_shared<UnityGPU>(1, 0.0f);
    }

    std::shared_ptr<UnityGPU> operator()(const std::vector<std::shared_ptr<UnityGPU>>& x) {
        // Forward: act = sum(w_i*x_i) + b
        std::shared_ptr<UnityGPU> act = std::make_shared<UnityGPU>(1, 0.0f);
        for (size_t i = 0; i < w.size(); i++) {
            auto prod = w[i]->mul(x[i]);
            auto tmp = act->add(prod);
            act = tmp;
        }
        act = act->add(b);

        // Non-linearity
        if (nonlin == 1) return act->relu();
        else if (nonlin == 2) return act->sigmoid();
        return act;
    }

    std::vector<std::shared_ptr<UnityGPU>> parameters() override {
        auto params = w;
        params.push_back(b);
        return params;
    }
};*/

__global__ void batch_matmul_kernel(const float* X, const float* W, const float* B,
    float* Y, int batch_size, int nin, int nout) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x; // colonne de sortie (nout)
    int by = blockIdx.y * blockDim.y + threadIdx.y; // ligne de batch (batch_size)
    if (bx < nout && by < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < nin; ++k)
            sum += X[by * nin + k] * W[k * nout + bx];
        Y[by * nout + bx] = sum + B[bx];
    }
}

// ---------------- Grad W ----------------
// Chaque thread écrit une case unique → pas besoin d'atomicAdd
__global__ void batch_matmul_backward_W_kernel(
    const float* __restrict__ X,
    const float* __restrict__ grad_Y,
    float* grad_W,
    int batch_size, int nin, int nout)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // input
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output

    if (col < nin && row < nout) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b)
            sum += X[b * nin + col] * grad_Y[b * nout + row];
        grad_W[col * nout + row] = sum;
    }
}

// ---------------- Grad B ----------------
// Plusieurs threads contribuent à la même case → atomicAdd
__global__ void batch_matmul_backward_B_kernel(
    const float* grad_Y, float* grad_B,
    int batch_size, int nout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nout) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b)
            sum += grad_Y[b * nout + idx];
        atomicAdd(&grad_B[idx], sum);
    }
}

// ---------------- Grad X ----------------
// Plusieurs sorties peuvent contribuer au même input → atomicAdd
__global__ void batch_matmul_backward_X_kernel(
    const float* grad_Y, const float* W, float* grad_X,
    int batch_size, int nin, int nout)
{
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch
    int i = blockIdx.x * blockDim.x + threadIdx.x; // input neuron

    if (b < batch_size && i < nin) {
        float sum = 0.0f;
        for (int j = 0; j < nout; ++j)
            sum += grad_Y[b * nout + j] * W[i * nout + j]; // W[i,j] row-major
        atomicAdd(&grad_X[b * nin + i], sum);
    }
}



class Node : public Module {
public:
    std::shared_ptr<UnityGPU> W;   // poids (nin x nout)
    std::shared_ptr<UnityGPU> B;   // biais (nout)
    int nin, nout;
    int nonlin; // 0=linear, 1=ReLU, 2=Sigmoid

    Node(int nin_, int nout_, int nonlin_ = 1) : nin(nin_), nout(nout_), nonlin(nonlin_) {
        // Initialisation W et B
        W = std::make_shared<UnityGPU>(nin * nout);
        B = std::make_shared<UnityGPU>(nout);

        // Init W aléatoire [-1,1]
        std::vector<float> hW(nin * nout);
        for (int i = 0; i < nin * nout; i++) hW[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        cudaMemcpy(W->data, hW.data(), nin * nout * sizeof(float), cudaMemcpyHostToDevice);

        // Biais à 0
        cudaMemset(B->data, 0, nout * sizeof(float));
    }

    // Forward batch: X = batch_size x nin
    std::shared_ptr<UnityGPU> forward_batch(std::shared_ptr<UnityGPU> X, int batch_size) {
        auto Y = std::make_shared<UnityGPU>(batch_size * nout);

        // Kernel pour multiplication batch_matmul(X, W) + B
        dim3 threads(16, 16);
        dim3 blocks((nout + threads.x - 1) / threads.x,
            (batch_size + threads.y - 1) / threads.y);

        batch_matmul_kernel << <blocks, threads >> > (
            X->data, W->data, B->data, Y->data,
            batch_size, nin, nout
            );

        // Appliquer activation
        int total = batch_size * nout;
        int threads_act = 256;
        int blocks_act = (total + threads_act - 1) / threads_act;

        if (nonlin == 1) relu_kernel << <blocks_act, threads_act >> > (Y->data, Y->data, total);
        else if (nonlin == 2) sigmoid_kernel << <blocks_act, threads_act >> > (Y->data, Y->data, total);

        // Backward autograd
        if (!NOGRAD) {
            Y->_prev = { X, W, B };
            Y->_backward = [this, X, Y, batch_size]() {
                // dY -> dX, dW, dB
                int threads = 16;
                int blocksX = (nin + threads - 1) / threads;
                int blocksY = (nout + threads - 1) / threads;

                batch_matmul_backward_X_kernel << <blocksX, threads >> > (
                    Y->grad, W->data, X->grad,
                    batch_size, nin, nout
                    );
                batch_matmul_backward_W_kernel << <blocksX, threads >> > (
                    X->data, Y->grad, W->grad,
                    batch_size, nin, nout
                    );
                batch_matmul_backward_B_kernel << <blocksY, threads >> > (
                    Y->grad, B->grad,
                    batch_size, nout
                    );
            };
        }

        return Y;
    }

    std::vector<std::shared_ptr<UnityGPU>> parameters() override {
        return { W, B };
    }
};





// ---------------- Layer ----------------
/*
class Layer : public Module {
public:
    std::vector<Node> nodes;

    Layer(int nin, int nout, int nonlin = 1) {
        for (int i = 0; i < nout; i++)
            nodes.emplace_back(nin, nonlin);
    }

    std::shared_ptr<UnityGPU> forward_batch(std::shared_ptr<UnityGPU> batch_inputs, int batch_size) {
        // Implémenter multiplication batch_inputs * weights + bias
        // Résultat dans un UnityGPU de taille batch_size * nout
        auto out = std::make_shared<UnityGPU>(batch_size * nodes.size());
        batch_matmul_kernel << <... >> > (batch_inputs->data, ..., out->data, batch_size, ...);
        // Appliquer nonlin via kernel GPU
        return out;
    }

    std::vector<std::shared_ptr<UnityGPU>> operator()(const std::vector<std::shared_ptr<UnityGPU>>& x) {
        std::vector<std::shared_ptr<UnityGPU>> out;
        for (auto& n : nodes)
            out.push_back(n(x));
        if (out.size() == 1) return { out[0] };
        return out;
    }

    std::vector<std::shared_ptr<UnityGPU>> parameters() override {
        std::vector<std::shared_ptr<UnityGPU>> params;
        for (auto& n : nodes) {
            auto p = n.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};*/




class Layer : public Module {
public:
    int nin, nout;
    int nonlin; // 0=lin, 1=relu, 2=sigmoid
    std::shared_ptr<UnityGPU> W; // [nin x nout]
    std::shared_ptr<UnityGPU> B; // [1 x nout]
    std::shared_ptr<UnityGPU> Y_cache;

    Layer(int nin_, int nout_, int nonlin_) : nin(nin_), nout(nout_), nonlin(nonlin_) {
        W = std::make_shared<UnityGPU>(nin * nout);
        B = std::make_shared<UnityGPU>(nout);
        // TODO: init W, B aléatoire sur GPU
        // Init aléatoire sur CPU puis copier vers GPU
        std::vector<float> hW(nin * nout);
        float scale = 1.0f / sqrt(nin);
        for (int i = 0; i < nin * nout; i++)
            hW[i] = ((float)rand() / RAND_MAX) * 2 * scale - scale;

        cudaMemcpy(W->data, hW.data(), nin * nout * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(B->data, 0, nout * sizeof(float));
    }

    // Forward batch GPU
    std::shared_ptr<UnityGPU> forward_batch(std::shared_ptr<UnityGPU> X, int batch_size) {
        //auto Y = std::make_shared<UnityGPU>(batch_size * nout);
        if (!Y_cache || Y_cache->size != batch_size * nout) {
            Y_cache = std::make_shared<UnityGPU>(batch_size * nout);
        }
        else {
            // reset gradients et valeurs
            cudaMemset(Y_cache->data, 0, Y_cache->size * sizeof(float));
            cudaMemset(Y_cache->grad, 0, Y_cache->size * sizeof(float));
        }
        auto Y = Y_cache; // alias

        dim3 threads(16, 16);
        dim3 blocks((nout + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);

        batch_matmul_kernel << <blocks, threads >> > (
            X->data, W->data, B->data, Y->data,
            batch_size, nin, nout
            );

        // Appliquer activation
        int N = batch_size * nout;
        int threads_act = 256;
        int blocks_act = (N + threads_act - 1) / threads_act;

        if (nonlin == 1)
            relu_kernel << <blocks_act, threads_act >> > (Y->data, Y->data, N);
        else if (nonlin == 2)
            sigmoid_kernel << <blocks_act, threads_act >> > (Y->data, Y->data, N);

        // Sauvegarde backward
        if (!NOGRAD) {
            Y->_prev = { X, W, B };
            Y->info = std::to_string(nin) + "/" + std::to_string(nout);

            Y->_backward = [this, X, Y, batch_size]() {
                int N = batch_size * nout;

                // --- Backward activation ---
                int threads_act = 256;
                int blocks_act = (N + threads_act - 1) / threads_act;

                if (nonlin == 1) { // ReLU
                    relu_backward_kernel << <blocks_act, threads_act >> > (Y->grad, Y->data, Y->grad, N);
                }
                else if (nonlin == 2) { // Sigmoid
                    sigmoid_backward_kernel << <blocks_act, threads_act >> > (Y->grad, Y->data, Y->grad, N);
                }

                // --- Backward weights ---
                dim3 threads(16, 16);
                dim3 blocks_W((nin + threads.x - 1) / threads.x, (nout + threads.y - 1) / threads.y);
                batch_matmul_backward_W_kernel << <blocks_W, threads >> > (
                    X->data, Y->grad, W->grad, batch_size, nin, nout
                    );

                // --- Backward bias ---
                int threads_B = 256;
                int blocks_B = (nout + threads_B - 1) / threads_B;
                batch_matmul_backward_B_kernel << <blocks_B, threads_B >> > (
                    Y->grad, B->grad, batch_size, nout
                    );

                // --- Backward input ---
                dim3 blocks_X((nin + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);
                batch_matmul_backward_X_kernel << <blocks_X, threads >> > (
                    Y->grad, W->data, X->grad, batch_size, nin, nout
                    );

                //std::cout << Y->info << std::endl;
                // --- Propagation vers les couches précédentes ---
                //if (X) X->backward();
            };
        }


        return Y;
    }

    /*std::shared_ptr<UnityGPU> forward_batch(std::shared_ptr<UnityGPU> X, int batch_size) {
        auto Y = std::make_shared<UnityGPU>(batch_size * nout);
        auto Z = std::make_shared<UnityGPU>(batch_size * nout); // pre-activation

        // --- Forward affine ---
        dim3 threads(16, 16);
        dim3 blocks((nout + threads.x - 1) / threads.x,
            (batch_size + threads.y - 1) / threads.y);

        batch_matmul_kernel << <blocks, threads >> > (
            X->data, W->data, B->data, Z->data,
            batch_size, nin, nout
            );

        // --- Forward activation ---
        int N = batch_size * nout;
        int threads_act = 256;
        int blocks_act = (N + threads_act - 1) / threads_act;

        if (nonlin == 1)
            relu_kernel << <blocks_act, threads_act >> > (Z->data, Y->data, N);
        else if (nonlin == 2)
            sigmoid_kernel << <blocks_act, threads_act >> > (Z->data, Y->data, N);
        else
            cudaMemcpy(Y->data, Z->data, N * sizeof(float), cudaMemcpyDeviceToDevice);

        // --- Préparation backward ---
        if (!NOGRAD) {
            Y->_prev = { X, W, B };
            Y->info = std::to_string(nin) + "/" + std::to_string(nout);

            // Stocker le pré-activation pour backward
            Y->_pre_activation = Z;

            Y->_backward = [this, X, Y, batch_size]() {
                int N = batch_size * nout;
                int threads_act = 256;
                int blocks_act = (N + threads_act - 1) / threads_act;

                // --- Backward activation ---
                if (nonlin == 1) { // ReLU
                    relu_backward_kernel << <blocks_act, threads_act >> > (
                        Y->grad, Y->_pre_activation->data, Y->grad, N
                        );
                }
                else if (nonlin == 2) { // Sigmoid
                    sigmoid_backward_kernel << <blocks_act, threads_act >> > (
                        Y->grad, Y->grad, Y->grad, N
                        );
                }

                // --- Backward poids W ---
                dim3 threads_W(16, 16);
                dim3 blocks_W((nin + threads_W.x - 1) / threads_W.x,
                    (nout + threads_W.y - 1) / threads_W.y);

                batch_matmul_backward_W_kernel << <blocks_W, threads_W >> > (
                    X->data, Y->grad, W->grad,
                    batch_size, nin, nout
                    );

                // --- Backward biais B ---
                int threads_B = 256;
                int blocks_B = (nout + threads_B - 1) / threads_B;
                batch_matmul_backward_B_kernel << <blocks_B, threads_B >> > (
                    Y->grad, B->grad, batch_size, nout
                    );

                // --- Backward input X ---
                dim3 blocks_X((nin + threads_W.x - 1) / threads_W.x,
                    (batch_size + threads_W.y - 1) / threads_W.y);
                batch_matmul_backward_X_kernel << <blocks_X, threads_W >> > (
                    Y->grad, W->data, X->grad,
                    batch_size, nin, nout
                    );
            };
        }

        return Y;
    }*/


    std::vector<std::shared_ptr<UnityGPU>> parameters() override {
        return { W, B };
    }


};


// ---------------- NN ----------------
class NN : public Module {
public:
    std::vector<Layer> layers;

    NN(int nin, const std::vector<int>& nouts, const std::vector<int>& nonlin) {
        for (size_t i = 0; i < nouts.size(); i++) {
            layers.emplace_back(i == 0 ? nin : nouts[i - 1], nouts[i], nonlin[i]);
        }
    }

    // Forward pour un batch entier (x: UnityGPU de shape batch_size x nin)
    std::shared_ptr<UnityGPU> forward_batch(std::shared_ptr<UnityGPU> x, int batch_size) {
        std::shared_ptr<UnityGPU> out = x;
        for (auto& layer : layers)
            out = layer.forward_batch(out, batch_size); // chaque layer doit aussi avoir forward_batch
        return out;
    }

    /*std::vector<std::shared_ptr<UnityGPU>> operator()(const std::vector<std::shared_ptr<UnityGPU>>& x) {
        std::vector<std::shared_ptr<UnityGPU>> out = x;
        for (auto& layer : layers)
            out = layer(out);
        return out;
    }*/

    std::vector<std::shared_ptr<UnityGPU>> parameters() override {
        std::vector<std::shared_ptr<UnityGPU>> params;
        for (auto& layer : layers) {
            auto p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};


// ---------------- MSE Loss GPU ----------------
// Kernel MSE pour GPU
__global__ void mse_loss_kernel(const float* pred, const float* target, float* grad, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / float(n);
    }
}

// Wrapper pour calculer le MSE et remplir grad dans chaque UnityGPU
void mse_loss_gpu(const std::vector<std::shared_ptr<UnityGPU>>& preds,
    const std::vector<std::shared_ptr<UnityGPU>>& targets)
{
    for (size_t i = 0; i < preds.size(); i++) {
        int n = preds[i]->size;
        float* d_pred = preds[i]->data;
        float* d_grad = preds[i]->grad;
        float* d_target = targets[i]->data; // target déjà sur GPU

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        mse_loss_kernel << <blocks, threads >> > (d_pred, d_target, d_grad, n);
    }
}

/*
void Training_GPU()
{
    srand(time(nullptr));

    // --- Paramètres réseau ---
    int nin = 2;
    std::vector<int> nouts = { 4, 1 };
    std::vector<int> nonlin = { 2, 2 }; // Sigmoid pour toutes les couches

    NN net(nin, nouts, nonlin);

    // --- Dataset XOR ---
    std::vector<std::pair<std::vector<float>, float>> dataset = {
        {{0,0},0}, {{0,1},1}, {{1,0},1}, {{1,1},0}
    };

    for (int i = 0; i < 500; ++i) {
        dataset.push_back(dataset[i % 4]);
    }

    float lr = 0.1f;
    int epochs = 5000;
    int batch_size = 128;

    // --- Création d'un buffer de gradients pour chaque paramètre ---
    for (auto& p : net.parameters())
        p->zero_grad(); // initialiser à 0

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t start = 0; start < dataset.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, dataset.size());

            // --- Préparer le batch ---
            std::vector<std::vector<std::shared_ptr<UnityGPU>>> batch_inputs;
            std::vector<std::shared_ptr<UnityGPU>> batch_targets;

            for (size_t i = start; i < end; ++i) {
                std::vector<std::shared_ptr<UnityGPU>> x;
                for (auto v : dataset[i].first)
                    x.push_back(std::make_shared<UnityGPU>(1, v)); // données sur GPU
                batch_inputs.push_back(x);

                auto y = std::make_shared<UnityGPU>(1, dataset[i].second);
                batch_targets.push_back(y);
            }

            // --- Forward pass ---
            std::vector<std::shared_ptr<UnityGPU>> batch_preds;
            for (size_t i = 0; i < batch_inputs.size(); ++i)
                batch_preds.push_back(net(batch_inputs[i])[0]);

            // --- Calcul MSE loss et grad sur GPU ---
            mse_loss_gpu(batch_preds, batch_targets);

            // --- Backward pass GPU ---
            for (auto& p : net.parameters())
                p->backward(); // chaque UnityGPU gère son _backward GPU

            // --- Update GPU ---
            for (auto& p : net.parameters())
                p->update(lr);
        }

        if (epoch % 500 == 0)
            std::cout << "Epoch " << epoch << "\n";
    }

     // --- Test final ---
    std::cout << "\n=== Test final ===\n";
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& sample = dataset[i];
        const std::vector<float>& x_vals = sample.first;
        float y_val = sample.second;

        std::vector<std::shared_ptr<UnityGPU>> x;
        for (auto v : x_vals)
            x.push_back(std::make_shared<UnityGPU>(1, v));

        auto pred = net(x);

        // Copier le résultat GPU -> CPU
        float h_out;
        cudaMemcpy(&h_out, pred[0]->data, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "[" << x_vals[0] << "," << x_vals[1] << "] -> " << h_out
            << " (attendu " << y_val << ")\n";
    }

}*/

void Training_GPU_Fast()
{
    srand(time(nullptr));

    const int nin = 2;
    std::vector<int> nouts = { 64, 64, 1 };
    std::vector<int> nonlin = { 1, 1, 2 }; // Sigmoid

    NN net(nin, nouts, nonlin);

    // Créer un moteur aléatoire
    std::random_device rd;
    std::mt19937 g(rd());

    // Dataset XOR
    std::vector<std::pair<std::vector<float>, float>> dataset = {
        {{0,0},0}, {{0,1},1}, {{1,0},1}, {{1,1},0}
    };

    
    float lr = 0.01;
    int epochs = 20000;
    const int batch_size = 4; // tout le dataset en batch pour l'exemple

    // Préparer le batch input et target sur GPU
    auto x_batch = std::make_shared<UnityGPU>(batch_size * nin);
    auto y_batch = std::make_shared<UnityGPU>(batch_size);

    float h_x[batch_size * nin];
    float h_y[batch_size];

    //std::shuffle(dataset.begin(), dataset.end(), g);
    for (int i = 0; i < batch_size; i++) {
        h_x[i * nin + 0] = dataset[i].first[0];
        h_x[i * nin + 1] = dataset[i].first[1];
        h_y[i] = dataset[i].second;
    }

    cudaMemcpy(x_batch->data, h_x, batch_size * nin * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_batch->data, h_y, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward batch
        /*if (epoch % 10000 == 0) {
            std::shuffle(dataset.begin(), dataset.end(), g);
            for (int i = 0; i < batch_size; i++) {
                h_x[i * nin + 0] = dataset[i].first[0];
                h_x[i * nin + 1] = dataset[i].first[1];
                h_y[i] = dataset[i].second;
            }

            cudaMemcpy(x_batch->data, h_x, batch_size * nin * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(y_batch->data, h_y, batch_size * sizeof(float), cudaMemcpyHostToDevice);
        }*/

        for (auto& p : net.parameters())
            p->zero_grad();
        
        auto y_pred = net.forward_batch(x_batch, batch_size);

        // Calcul de la loss et des gradients sur GPU
        mse_loss_kernel << <1, batch_size >> > (y_pred->data, y_batch->data, y_pred->grad, batch_size);
        //mse_loss_gpu(y_pred, y_batch);
              
        y_pred->backward(); // propage les gradients sur tous les paramètres
        //for (auto& p : net.parameters())
        //    p->_backward();

        // Update params
        for (auto& p : net.parameters())
            p->update(lr);

        if (epoch % 500 == 0) {
            // Lire la loss GPU pour affichage
            float h_pred[batch_size];
            cudaMemcpy(h_pred, y_pred->data, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
            float loss = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                float diff = h_pred[i] - h_y[i];
                loss += diff * diff;
            }
            std::cout << "Epoch " << epoch << ", loss=" << loss << "\n";
        }
    }

    // Test final
    float h_pred[4];
    auto y_pred = net.forward_batch(x_batch, 4);
    cudaMemcpy(h_pred, y_pred->data, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << "Input: [" << h_x[i * nin] << "," << h_x[i * nin + 1] << "] -> "
            << h_pred[i] << " (expected " << h_y[i] << ")\n";
    }

}



int main()
{
 
    Training_GPU_Fast();
  
    return 0;
}
