using LinearAlgebra 

struct ZeroStepsizeError <: Exception end

function compute_gradient(func, x)
    n_dim = length(x)

    fx = func(x)
    dx = 1e-10

    grad = zeros(n_dim)
    for i in 1:n_dim
        x_ = copy(x)
        x_[i] += dx
        grad[i] = (func(x_) - fx)/dx
    end
    return grad
end

function line_search(d, func, x)
    xi = 1e-4 
    tau = 0.87
    grad = compute_gradient(func, x)
    predicate(alpha) = (func(x + alpha*d) < func(x) + xi * alpha * grad' * d) 

    alpha = 1.0
    while(~predicate(alpha))
        alpha *= tau
        alpha == 0.0 && throw(ZeroStepsizeError)
    end
    return alpha
end

function update_gradient_descent(func, x)
    d = -normalize(compute_gradient(func, x))
    alpha = line_search(d, func, x)
    x += alpha * d
    return x
end

function gen_update_nesterov(L)
    x_pre = nothing
    k = 1 
    beta(k) = k/(k+3.)
    function update_nesterov(func, x) 
        # handle init case
        x_pre == nothing && (x_pre = x)

        # update rule
        x_new_inertial = x + beta(k-1) * (x - x_pre)
        grad = compute_gradient(func, x_new_inertial)
        x_new = x_new_inertial - (1.0/L) * grad
        
        # update inner state
        x_pre = x
        k += 1
        return x_new
    end
    return update_nesterov
end

function solve(func, x_init, x_optimial, update_method; N_itr=10000)
    f_seq = zeros(N_itr)
    time_seq = zeros(N_itr)
    
    f_optimal = func(x_optimial)
    x = x_init
    for i in 1:N_itr
        x = update_method(func, x)

        # result
        f_now = func(x)
        f_seq[i] = f_now - f_optimal
        time_seq[i] = time()

        # print
        println(i)
        err = abs(f_optimal - f_now)
        println("err: " * string(err))
    end
    return f_seq, time_seq
end

function post_process(err_seq)
    err_seq_reversed = reverse(err_seq)
    err_max_seq_reversed = []
    err_max = -Inf
    for err in err_seq_reversed 
        err_max = max(err_max, err)
        push!(err_max_seq_reversed, err_max)
    end
    return reverse(err_max_seq_reversed)
end

using Random
import JSON
Random.seed!(1)

function single_run(n, nesterov)
    A = rand(n, n)
    w = randn(n, 1)
    b = A * w + randn(n, 1) * 0.1
    w_sol = inv(A'*A)*A'*b

    f(x) = norm(b - A*x)^2
    w_init = randn(n, 1)

    L = norm(compute_gradient(f, w_init))/norm(w_sol - w_init) * n
    update_rule = nesterov ? gen_update_nesterov(L) : update_gradient_descent
    err_seq, time_seq = solve(f, w_init, w_sol, update_rule; N_itr=30000)
    err_seq_processed = post_process(err_seq)
    data = Dict("err_seq"=>err_seq, "time_seq"=>time_seq, "err_seq_processed"=>err_seq_processed)
    return data
end

nesterov = false
for nesterov in [true, false]
    for n in [4, 32, 128]
        data = single_run(n, nesterov)
        filename = "json/n"* string(n) * "_" * (nesterov ? "nesterov" : "grad") * ".json"
        open(filename, "w") do f
            JSON.print(f, data)
        end
    end
end
