// RMGRADAVX.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <set>
#include <functional>
#include <memory>
#include <string>
#include <chrono>
#include <unordered_set>

using namespace std;

bool NOGRAD = false;  // variable globale

struct NoGradGuard {
    bool old;
    NoGradGuard() {
        old = NOGRAD;
        NOGRAD = true;
    }
    ~NoGradGuard() {
        NOGRAD = old;
    }
};

/*
class Unity : public std::enable_shared_from_this<Unity> {
public:
    double unity;                                   // valeur
    double grad;                                    // gradient
    std::vector<std::shared_ptr<Unity>> _prev;      // dépendances
    std::function<void()> _backward;               // fonction backward

    // Constructeur
    Unity(double value, const std::vector<std::shared_ptr<Unity>>& children = {})
        : unity(value), grad(0.0)
    {
        if (NOGRAD) {
            _prev.clear();
            _backward = []() {};
        }
        else {
            _prev = children;
            _backward = []() {};
        }
    }

    // ---------------- Unary ----------------
    std::shared_ptr<Unity> neg() {
        return mul(std::make_shared<Unity>(-1.0));
    }

    // ---------------- Binary Unity ----------------
    std::shared_ptr<Unity> add(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(unity + other->unity, std::vector{ shared_from_this(), other });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += out->grad;
                other->grad += out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sub(const std::shared_ptr<Unity>& other) {
        return add(other->neg());
    }

    std::shared_ptr<Unity> mul(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(unity * other->unity, std::vector{ shared_from_this(), other });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += other->unity * out->grad;
                other->grad += self->unity * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> div(const std::shared_ptr<Unity>& other) {
        return mul(other->pow(-1.0));
    }

    // ---------------- Binary double ----------------
    std::shared_ptr<Unity> add(double val) { return add(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> sub(double val) { return sub(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> mul(double val) { return mul(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> div(double val) { return div(std::make_shared<Unity>(val)); }

    // ---------------- Pow ----------------
    std::shared_ptr<Unity> pow(double other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other), std::vector{ shared_from_this() });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other * std::pow(self->unity, other - 1.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> pow(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other->unity), std::vector{ shared_from_this(), other });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other->unity * std::pow(self->unity, other->unity - 1.0) * out->grad;
                if (self->unity > 0.0)
                    other->grad += std::log(self->unity) * out->unity * out->grad;
            };
        }
        return out;
    }

    // ---------------- Activations ----------------
    std::shared_ptr<Unity> relu() {
        auto out = std::make_shared<Unity>(unity < 0.0 ? 0.0 : unity, std::vector{ shared_from_this() });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), out]() {
                self->grad += (out->unity > 0.0 ? 1.0 : 0.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sigmoid() {
        double s = 1.0 / (1.0 + std::exp(-unity));
        auto out = std::make_shared<Unity>(s, std::vector{ shared_from_this() });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), s, out]() {
                self->grad += s * (1.0 - s) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> tanh() {
        double t = std::tanh(unity);
        auto out = std::make_shared<Unity>(t, std::vector{ shared_from_this() });
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), t, out]() {
                self->grad += (1.0 - t * t) * out->grad;
            };
        }
        return out;
    }

    // ---------------- Backward ----------------
    void backward() {
        std::vector<std::shared_ptr<Unity>> topo;
        std::set<std::shared_ptr<Unity>> visited;

        std::function<void(std::shared_ptr<Unity>)> build_topo = [&](std::shared_ptr<Unity> v) {
            if (!v) return;
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& child : v->_prev) build_topo(child);
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it) (*it)->_backward();
        }
    }

    // ---------------- Print ----------------
    void print() const {
        std::cout << "Unity(unity=" << unity << ", grad=" << grad << ")\n";
    }
};*/

/*
class Unity : public std::enable_shared_from_this<Unity> {
public:
    double unity;   // valeur
    double grad;    // gradient
    std::vector<std::shared_ptr<Unity>> _prev;
    std::function<void()> _backward;

    Unity(double value, const std::vector<std::shared_ptr<Unity>>& children = {})
        : unity(value), grad(0.0)
    {
        if (NOGRAD) {
            _prev.clear();
            _backward = []() {};
        }
        else {
            _prev = children;
            _backward = []() {};
        }
    }

    // ---------------- Unary ----------------
    std::shared_ptr<Unity> neg() {
        return mul(-1.0);
    }

    // ---------------- Binary Unity ----------------
    std::shared_ptr<Unity> add(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(
            unity + other->unity,
            std::vector<std::shared_ptr<Unity>>{ shared_from_this(), other }
        );

        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += out->grad;
                other->grad += out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sub(const std::shared_ptr<Unity>& other) {
        return add(other->neg());
    }

    std::shared_ptr<Unity> mul(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(
            unity * other->unity,
            std::vector<std::shared_ptr<Unity>>{ shared_from_this(), other }
        );
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += other->unity * out->grad;
                other->grad += self->unity * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> div(const std::shared_ptr<Unity>& other) {
        return mul(other->pow(-1.0));
    }

    // ---------------- Binary double ----------------
    std::shared_ptr<Unity> add(double val) { return add(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> sub(double val) { return sub(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> mul(double val) { return mul(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> div(double val) { return div(std::make_shared<Unity>(val)); }

    // ---------------- Pow ----------------
    std::shared_ptr<Unity> pow(double other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other), std::vector<std::shared_ptr<Unity>>{shared_from_this()});
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other * std::pow(self->unity, other - 1.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> pow(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other->unity), std::vector<std::shared_ptr<Unity>>{shared_from_this(), other});
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other->unity * std::pow(self->unity, other->unity - 1.0) * out->grad;
                if (self->unity > 0.0)
                    other->grad += std::log(self->unity) * out->unity * out->grad;
            };
        }
        return out;
    }

    // ---------------- Activations ----------------
    std::shared_ptr<Unity> relu() {
        double val = unity < 0.0 ? 0.0 : unity;
        auto out = std::make_shared<Unity>(val, std::vector<std::shared_ptr<Unity>>{shared_from_this()});
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), out]() {
                self->grad += (out->unity > 0.0 ? 1.0 : 0.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sigmoid() {
        double s = 1.0 / (1.0 + std::exp(-unity));
        auto out = std::make_shared<Unity>(s, std::vector<std::shared_ptr<Unity>>{shared_from_this()});
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), s, out]() {
                self->grad += s * (1.0 - s) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> tanh() {
        double t = std::tanh(unity);
        auto out = std::make_shared<Unity>(t, std::vector<std::shared_ptr<Unity>>{shared_from_this()});
        if (!NOGRAD) {
            out->_backward = [self = shared_from_this(), t, out]() {
                self->grad += (1.0 - t * t) * out->grad;
            };
        }
        return out;
    }

    // ---------------- Backward ----------------
    void backward() {
        std::vector<std::shared_ptr<Unity>> topo;
        std::set<std::shared_ptr<Unity>> visited;

        std::function<void(std::shared_ptr<Unity>)> build_topo = [&](std::shared_ptr<Unity> v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& child : v->_prev)
                    if (child) build_topo(child);
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it) (*it)->_backward();
        }
    }

    void backward() {
        std::vector<std::shared_ptr<Unity>> topo;
        std::unordered_set<Unity*> visited;  // on stocke les pointeurs bruts

        // Fonction récursive pour construire le topo
        std::function<void(std::shared_ptr<Unity>)> build_topo = [&](std::shared_ptr<Unity> v) {
            if (!v) return;
            if (visited.find(v.get()) == visited.end()) {
                visited.insert(v.get());
                for (auto& child : v->_prev)
                    if (child) build_topo(child);
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it) (*it)->_backward();
        }
    }


    void print() const {
        std::cout << "Unity(unity=" << unity << ", grad=" << grad << ")" << std::endl;
    }
};*/

class Unity : public std::enable_shared_from_this<Unity> {
public:
    double unity;   // valeur
    double grad;    // gradient
    std::vector<std::weak_ptr<Unity>> _prev; // parents faibles pour éviter les cycles
    std::function<void()> _backward;

    Unity(double value) : unity(value), grad(0.0) {
        _backward = []() {};
    }

    // ---------------- Unary ----------------
    std::shared_ptr<Unity> neg() {
        return mul(-1.0);
    }

    // ---------------- Binary Unity ----------------
    std::shared_ptr<Unity> add(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(unity + other->unity);
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_prev.push_back(other);
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += out->grad;
                other->grad += out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sub(const std::shared_ptr<Unity>& other) {
        return add(other->neg());
    }

    std::shared_ptr<Unity> mul(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(unity * other->unity);
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_prev.push_back(other);
            out->_backward = [self = shared_from_this(), other, out]() {
                self->grad += other->unity * out->grad;
                other->grad += self->unity * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> div(const std::shared_ptr<Unity>& other) {
        return mul(other->pow(-1.0));
    }

    // ---------------- Binary double ----------------
    std::shared_ptr<Unity> add(double val) { return add(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> sub(double val) { return sub(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> mul(double val) { return mul(std::make_shared<Unity>(val)); }
    std::shared_ptr<Unity> div(double val) { return div(std::make_shared<Unity>(val)); }

    // ---------------- Pow ----------------
    std::shared_ptr<Unity> pow(double other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other));
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other * std::pow(self->unity, other - 1.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> pow(const std::shared_ptr<Unity>& other) {
        auto out = std::make_shared<Unity>(std::pow(unity, other->unity));
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_prev.push_back(other);
            out->_backward = [self = shared_from_this(), other, out]() {
                if (self->unity != 0.0)
                    self->grad += other->unity * std::pow(self->unity, other->unity - 1.0) * out->grad;
                if (self->unity > 0.0)
                    other->grad += std::log(self->unity) * out->unity * out->grad;
            };
        }
        return out;
    }

    // ---------------- Activations ----------------
    std::shared_ptr<Unity> relu() {
        double val = unity < 0.0 ? 0.0 : unity;
        auto out = std::make_shared<Unity>(val);
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_backward = [self = shared_from_this(), out]() {
                self->grad += (out->unity > 0.0 ? 1.0 : 0.0) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> sigmoid() {
        double s = 1.0 / (1.0 + std::exp(-unity));
        auto out = std::make_shared<Unity>(s);
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_backward = [self = shared_from_this(), s, out]() {
                self->grad += s * (1.0 - s) * out->grad;
            };
        }
        return out;
    }

    std::shared_ptr<Unity> tanh() {
        double t = std::tanh(unity);
        auto out = std::make_shared<Unity>(t);
        if (!NOGRAD) {
            out->_prev.push_back(shared_from_this());
            out->_backward = [self = shared_from_this(), t, out]() {
                self->grad += (1.0 - t * t) * out->grad;
            };
        }
        return out;
    }

    // ---------------- Backward ----------------
    void backward() {
        std::vector<std::shared_ptr<Unity>> topo;
        std::unordered_set<Unity*> visited;

        std::function<void(std::shared_ptr<Unity>)> build_topo = [&](std::shared_ptr<Unity> v) {
            if (!v) return;
            if (visited.find(v.get()) != visited.end()) return;
            visited.insert(v.get());
            for (auto& child_weak : v->_prev) {
                if (auto child = child_weak.lock()) {
                    build_topo(child);
                }
            }
            topo.push_back(v);
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it && (*it)->_backward) (*it)->_backward();
        }
    }

    void print() const {
        std::cout << "Unity(unity=" << unity << ", grad=" << grad << ")" << std::endl;
    }
};


class Module {
public:
    // Retourne les paramètres du module : à surcharger dans les classes dérivées
    virtual std::vector<std::shared_ptr<Unity>> parameters() {
        return {};
    }

    // Met tous les gradients à zéro
    void zero_grad() {
        for (auto& p : parameters()) {
            if (p) p->grad = 0.0;
        }
    }

    virtual ~Module() = default;
};

#define AVX_NODE256

#ifdef AVX_NODE256D

class Node : public Module {
public:
    std::vector<std::shared_ptr<Unity>> w;
    std::shared_ptr<Unity> b;
    int nonlin;

    Node(int nin, int nonlin_ = 0) : nonlin(nonlin_) {
        for (int i = 0; i < nin; i++)
            w.push_back(std::make_shared<Unity>((rand() / (double)RAND_MAX) * 2 - 1));
        b = std::make_shared<Unity>(0.0);
    }

    std::shared_ptr<Unity> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        int n = (int)w.size();
        __m256d sum_vec = _mm256_setzero_pd();
        double sum_arr[4] = { 0 };

        int i = 0;
        for (; i + 3 < n; i += 4) {
            __m256d vw = _mm256_set_pd(w[i]->unity, w[i + 1]->unity, w[i + 2]->unity, w[i + 3]->unity);
            __m256d vx = _mm256_set_pd(x[i]->unity, x[i + 1]->unity, x[i + 2]->unity, x[i + 3]->unity);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vw, vx));
        }

        _mm256_storeu_pd(sum_arr, sum_vec);
        double total = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

        for (; i < n; i++) total += w[i]->unity * x[i]->unity;

        auto act = std::make_shared<Unity>(total + b->unity);

        if (!NOGRAD) {
            for (auto& wi : w) act->_prev.push_back(wi);
            for (auto& xi : x) act->_prev.push_back(xi);
            act->_prev.push_back(b);

            act->_backward = [act, x, this_w = w, b_ = b]() mutable {
                for (size_t j = 0; j < this_w.size(); j++)
                    this_w[j]->grad += x[j]->unity * act->grad;
                for (size_t j = 0; j < x.size(); j++)
                    x[j]->grad += this_w[j]->unity * act->grad;
                b_->grad += act->grad;
            };
        }

        // Apply non-linearity
        if (nonlin == 0) return act;
        else if (nonlin == 1) return act->relu();
        else if (nonlin == 2) return act->sigmoid();
        else if (nonlin == 3) return act->tanh();
        return act;
    }

    std::vector<std::shared_ptr<Unity>> parameters() {
        auto params = w;
        params.push_back(b);
        return params;
    }
};

#endif

#define OM
#ifdef AVX_NODE256
class Node : public Module {
public:
    std::vector<std::shared_ptr<Unity>> w;
    std::shared_ptr<Unity> b;
    int nonlin;

    Node(int nin, int nonlin_ = 0) : nonlin(nonlin_) {
        for (int i = 0; i < nin; i++) {
            // stocker en float dans Unity si tu veux maximiser le gain
            w.push_back(std::make_shared<Unity>((float)rand() / RAND_MAX * 2.0f - 1.0f));
        }
        b = std::make_shared<Unity>((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

#ifndef OMPX
    std::shared_ptr<Unity> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        int n = (int)w.size();
        __m256 sum_vec = _mm256_setzero_ps(); // 8 floats
        float sum_arr[8] = { 0 };

        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vw = _mm256_set_ps(
                (float)w[i]->unity, (float)w[i + 1]->unity, (float)w[i + 2]->unity, (float)w[i + 3]->unity,
                (float)w[i + 4]->unity, (float)w[i + 5]->unity, (float)w[i + 6]->unity, (float)w[i + 7]->unity
            );
            __m256 vx = _mm256_set_ps(
                (float)x[i]->unity, (float)x[i + 1]->unity, (float)x[i + 2]->unity, (float)x[i + 3]->unity,
                (float)x[i + 4]->unity, (float)x[i + 5]->unity, (float)x[i + 6]->unity, (float)x[i + 7]->unity
            );
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vw, vx));
        }

        _mm256_storeu_ps(sum_arr, sum_vec);
        float total = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
            sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

        for (; i < n; i++) total += (float)(w[i]->unity * x[i]->unity);

        auto act = std::make_shared<Unity>(total + b->unity);

        if (!NOGRAD) {
            // Ajouter les dépendances en weak_ptr pour éviter le cycle
            for (auto& wi : w) act->_prev.push_back(std::weak_ptr<Unity>(wi));
            for (auto& xi : x) act->_prev.push_back(std::weak_ptr<Unity>(xi));
            act->_prev.push_back(std::weak_ptr<Unity>(b));

            // Définir le backward en capturant act faiblement
            std::weak_ptr<Unity> act_wp = act;
            act->_backward = [act_wp, x, this]() {
                if (auto act_sp = act_wp.lock()) {  // récupère le shared_ptr valide
                    for (size_t j = 0; j < w.size(); j++)
                        w[j]->grad += x[j]->unity * act_sp->grad;  // gradient par rapport aux poids
                    for (size_t j = 0; j < x.size(); j++)
                        x[j]->grad += w[j]->unity * act_sp->grad;  // gradient par rapport aux entrées
                    b->grad += act_sp->grad;                        // gradient par rapport au biais
                }
            };
        }


        // Activation
        if (nonlin == 0) return act;
        else if (nonlin == 1) return act->relu();
        else if (nonlin == 2) return act->sigmoid();
        else if (nonlin == 3) return act->tanh();
        return act;
    }
#endif

#ifdef OMPX

    std::shared_ptr<Unity> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        int n = (int)w.size();
        float total = 0.0f;  // déclaration AVANT le parallel

#pragma omp parallel for reduction(+:total)
        for (int i = 0; i < n; i += 8) {
            float local_sum = 0.0f;
            if (i + 7 < n) {
                __m256 vw = _mm256_set_ps(
                    (float)w[i]->unity, (float)w[i + 1]->unity,
                    (float)w[i + 2]->unity, (float)w[i + 3]->unity,
                    (float)w[i + 4]->unity, (float)w[i + 5]->unity,
                    (float)w[i + 6]->unity, (float)w[i + 7]->unity
                );
                __m256 vx = _mm256_set_ps(
                    (float)x[i]->unity, (float)x[i + 1]->unity,
                    (float)x[i + 2]->unity, (float)x[i + 3]->unity,
                    (float)x[i + 4]->unity, (float)x[i + 5]->unity,
                    (float)x[i + 6]->unity, (float)x[i + 7]->unity
                );
                __m256 sum_vec = _mm256_mul_ps(vw, vx);
                float sum_arr[8];
                _mm256_storeu_ps(sum_arr, sum_vec);
                for (int k = 0; k < 8; k++) local_sum += sum_arr[k];
            }
            else {
                // reste pour n % 8
                for (int j = i; j < n; j++) local_sum += (float)(w[j]->unity * x[j]->unity);
            }

            total += local_sum;  // reduction gère la somme globale
        }

        auto act = std::make_shared<Unity>(total + b->unity);

        if (!NOGRAD) {
            for (auto& wi : w) act->_prev.push_back(wi);
            for (auto& xi : x) act->_prev.push_back(xi);
            act->_prev.push_back(b);

            act->_backward = [act, x, this_w = w, b_ = b]() mutable {
                for (size_t j = 0; j < this_w.size(); j++)
                    this_w[j]->grad += x[j]->unity * act->grad;
                for (size_t j = 0; j < x.size(); j++)
                    x[j]->grad += this_w[j]->unity * act->grad;
                b_->grad += act->grad;
            };
        }

        // Activation
        if (nonlin == 0) return act;
        else if (nonlin == 1) return act->relu();
        else if (nonlin == 2) return act->sigmoid();
        else if (nonlin == 3) return act->tanh();
        return act;
    }


#endif

    std::vector<std::shared_ptr<Unity>> parameters() {
        auto params = w;
        params.push_back(b);
        return params;
    }
};

#endif

#ifdef NODE
class Node : public Module {
public:
    std::vector<std::shared_ptr<Unity>> w;
    std::shared_ptr<Unity> b;
    int nonlin; // 0 = Linear, 1 = ReLU, 2 = Sigmoid

    Node(int nin, int nonlin_ = 0) : nonlin(nonlin_) {
        for (int i = 0; i < nin; i++) {
            double r = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
            w.push_back(std::make_shared<Unity>(r));
        }
        b = std::make_shared<Unity>(0.0);
    }

    std::shared_ptr<Unity> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        // act = sum(wi * xi) + b
        std::shared_ptr<Unity> act = b;
        for (size_t i = 0; i < w.size(); i++) {
            act = act->add(w[i]->mul(x[i]));
        }


        // Apply non-linearity
        if (nonlin == 0) return act;
        else if (nonlin == 1) return act->relu();
        else if (nonlin == 2) return act->sigmoid();
        else if (nonlin == 3) return act->tanh();
        return act;
      
    }

    std::vector<std::shared_ptr<Unity>> parameters() override {
        std::vector<std::shared_ptr<Unity>> params = w;
        params.push_back(b);
        return params;
    }

    std::string repr() const {
        std::string type = (nonlin == 0 ? "Linear" : (nonlin == 1 ? "ReLU" : "Sigmoid"));
        return type + "Node(" + std::to_string(w.size()) + ")";
    }
};
#endif


class Layer : public Module {
public:
    std::vector<Node> nodes;

    Layer(int nin, int nout, int nonlin = 0) {
        for (int i = 0; i < nout; i++)
            nodes.emplace_back(nin, nonlin);
    }

    std::vector<std::shared_ptr<Unity>> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        std::vector<std::shared_ptr<Unity>> out;
        for (auto& n : nodes)
            out.push_back(n(x));
        return out;
    }

    std::vector<std::shared_ptr<Unity>> parameters() {
        std::vector<std::shared_ptr<Unity>> params;
        for (auto& n : nodes) {
            auto p = n.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

class NN : public Module {
public:
    std::vector<Layer> layers;

    NN(int nin, const std::vector<int>& nouts, const std::vector<int>& nonlin) {
        for (size_t i = 0; i < nouts.size(); i++)
            layers.emplace_back(i == 0 ? nin : nouts[i - 1], nouts[i], nonlin[i]);
    }

    std::vector<std::shared_ptr<Unity>> operator()(const std::vector<std::shared_ptr<Unity>>& x) {
        std::vector<std::shared_ptr<Unity>> out = x;
        for (auto& layer : layers)
            out = layer(out);
        return out;
    }

    std::vector<std::shared_ptr<Unity>> parameters() {
        std::vector<std::shared_ptr<Unity>> params;
        for (auto& layer : layers) {
            auto p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

// ---------------- Loss ----------------
std::shared_ptr<Unity> mse_loss(const std::shared_ptr<Unity>& pred, const std::shared_ptr<Unity>& target) {
    return pred->sub(target)->pow(2.0);
}
// ---------------- Training ----------------
void Training() {
    auto start = std::chrono::high_resolution_clock::now();

    NN net(2, { 4,1 }, { 2,2 }); // 2->4->1

    std::vector<std::pair<std::vector<double>, double>> unity = {
        {{0,0},0}, {{0,1},1}, {{1,0},1}, {{1,1},0}
    };

    double lr = 0.1;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (auto& [x_vals, y_val] : unity) {
            std::vector<std::shared_ptr<Unity>> x_u;
            for (auto v : x_vals) x_u.push_back(std::make_shared<Unity>(v));
            auto y_u = std::make_shared<Unity>(y_val);

            auto pred = net(x_u);
            auto loss = mse_loss(pred[0], y_u);

            // backward
            //for (auto& p : net.parameters()) p->grad = 0.0;
            net.zero_grad();
            loss->backward();

            // update
            for (auto& p : net.parameters()) p->unity -= lr * p->grad;

            total_loss += loss->unity;
        }

        if (epoch % 500 == 0) std::cout << "Epoch " << epoch << ", loss=" << total_loss << "\n";
    }

    // test
    for (auto& [x_vals, y_val] : unity) {
        std::vector<std::shared_ptr<Unity>> x_u;
        for (auto v : x_vals) x_u.push_back(std::make_shared<Unity>(v));
        auto pred = net(x_u);
        std::cout << "[" << x_vals[0] << "," << x_vals[1] << "] -> " << pred[0]->unity
            << " (expected " << y_val << ")\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Training time: " << elapsed.count() << " seconds\n";

}


int main()
{
    auto x = std::make_shared<Unity>(2.0);
    auto y = std::make_shared<Unity>(3.0);

    // construire l'expression avec shared_ptr
    auto z = x->mul(y)
        ->add(x->div(y))
        ->sub(x->pow(2.0))
        ->add(x->relu())
        ->add(x->sigmoid())
        ->add(x->tanh())
        ->sub(y);

    z->backward();

    x->print();   // affichera valeur et grad
    y->print();
    z->print();

    Training();
   
}

