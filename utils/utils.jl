using LinearAlgebra
using Test
using Base

# Captures problem instances
abstract type AbstractVI end

Base.@kwdef struct VariationalInequality <: AbstractVI
    dim::Int64
    F::Function
    Π::Function
    H = identity
    sol = missing
    xdim::Int64

    # constructor
    function VariationalInequality(F::Function, Π::Function; H=missing, sol=missing,
                                   xdim::Union{Missing,Int64}=missing, dim::Union{Missing,Int64}=missing)
        # if not specified assume that xdim==ydim
        if !ismissing(sol)
            dim = size(sol)[1]
        elseif ismissing(dim)
            err("If no solution is given, dimension needs to be specified.")
            # I could combine ProblemParams and VI into one thing then one could deduce dim from z0
        end
        if ismissing(xdim)
            xdim = Int(dim/2)
        end

        return new(dim, F, Π, H, sol, xdim)
    end
end


function mk_folder_structure(folder_name)
    path = "../output/$folder_name"
    mkpath(path)
    return path
end


function simplex_projection(x)
    """Projection onto the unit simplex."""

    u = sort(x, rev=true)
    ρ = 1
    i = 1
    for i in 1:length(x)
        if u[i] + 1/i * (1 - sum(u[1:i])) > 0
            ρ = i
        end
    end
    λ = 1/ρ * (1 - sum(u[1:ρ]))
    out = max.(x .+ λ, 0)
    return out
end


function proj_twice_simplex(z)
    "Projection onto twice the unit simplex, i.e. Δ × Δ. "
    # TODO: this assumes that xdim == dim/2 which I do not assume anywhere else
    n = Int(length(z)/2)
    return [simplex_projection(z[1:n]); simplex_projection(z[n+1:end])]
end


Base.@kwdef struct Callback
    """Can be passed to an Algorithm and is supposed to be called after every iteration to store
     various optimality measures. """
    label::String
    x_ticks::Vector{Float64} = []
    iterates::Array = []
    store_grad::Vector{Float64} = []
    store_hamiltonian_grad::Vector{Float64} = []
    vi_val::Vector{Float64} = []
    pd_gap::Vector{Float64} = []
    lr::Vector{Float64} = []
    lr_ratio::Vector{Float64} = []
end

function (cb::Callback)(VI::AbstractVI, u, lr, Fu, lr_ratio, k; Hu=missing)
    "Makes the Callback struct actually callable and computes various optimality measures "
    append!(cb.store_grad, compute_norm_grad(VI, u, lr, Fu))
    if !ismissing(VI.sol)
        append!(cb.vi_val, -2*eval_vi(VI, u, Fu)/norm(Fu)^2/lr)
    end
    append!(cb.pd_gap, compute_gap(VI, u))
    append!(cb.x_ticks, k)
    if VI.dim <= 4
        append!(cb.iterates, u)
    end
    if !ismissing(Hu)
        append!(cb.store_hamiltonian_grad, norm(Hu)^2)
    elseif !ismissing(VI.H)
        append!(cb.store_hamiltonian_grad, norm(VI.H(u))^2)
    end
    append!(cb.lr, lr)
    append!(cb.lr_ratio, lr_ratio)
end

function compute_norm_grad(VI::AbstractVI, u, lr, Fu=missing)
    """Compute fixed point residual."""
    if ismissing(Fu); Fu = VI.F(u) end
    Π = VI.Π
    return norm(Π(u - lr*Fu) - u)^2/lr^2
end

function eval_vi(VI::AbstractVI, u, Fu=missing)
    """Computes a quantity that should be positive for monotone problems."""
    if ismissing(Fu); Fu = VI.F(u) end
    sol = VI.sol
    return Fu' * (u - sol)
end


struct MatrixGame <: AbstractVI
    dim::Int64
    # same as VI the matrix A defines the F
    A::Matrix{Float64}
    Π::Function
    sol
    # do not set this directly!
    F::Function
    H::Function
    xdim::Int64

    # constructor
    function MatrixGame(A::Matrix{Float64}, Π::Function; sol=missing, xdim=0.)
        dim = sum(size(A))
        xdim = size(A)[2]
        F(u) = [A' * u[xdim+1:end]; - A * u[1:xdim]]
        # ydim = dim-xdim
        # J(u) = [zeros(xdim, xdim) A'; zeros(ydim, ydim)]
        function H(u)
            "Hamiltonian"
            x, y = u[1:xdim], u[xdim+1:end]
            return [A'*A*x; A*A'*y]
        end

        return new(dim, A, Π, sol, F, H, xdim)
    end
end

function compute_gap(game::AbstractVI, u)
    """In general we do not know how to compute the gap"""
    return []
end


function compute_gap(game::MatrixGame, u)
    """Compute primal dual Gap for a matrix game at point u."""
    n = game.xdim
    A = game.A
    maximum(A*u[1:n]) - minimum(A'*u[n+1:end])
end


Base.@kwdef mutable struct ProblemParams
    z0::Array{Float64}
    n_grad_eval::Int64
    path::String
    L = Inf
    # (estimate of) weak Minty parameter
    ρ = missing

    function ProblemParams(z0::Array{Float64}, n_grad_eval::Int64, folder_name::String, L=Inf; ρ=missing)
        path = mk_folder_structure(folder_name)
        return new(z0, n_grad_eval, path, L, ρ)
    end
end

Base.@kwdef struct AlgParams
    γ::Float64
    α::Float64
    lr::Float64
    ϕ::Float64
    callback::Callback
end

mutable struct algorithm
    method::Function
    label::String
    params::NamedTuple
    # constructor
    function algorithm(method::Function, label::String, params=(;))
        return new(method, label, params)
    end
end
