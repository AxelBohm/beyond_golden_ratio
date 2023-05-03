# For all the adaptive methods we use the step size parameter only to compute an initial forward step and then use this new point to estimate the local lipschitz constant
using LinearAlgebra: norm

include("utils.jl")


function eg(VI::AbstractVI, params::ProblemParams, cb::Callback; γ::Float64=1., adaptive=false, universal=false, anchoring=missing, ada_heuristic=false)
    """Extra-gradient method for VI given by F with option for adaptive step size."""

    F, Π = VI.F, VI.Π
    z = z0 = params.z0
    lr = 1/params.L

    if adaptive
        # more practical way to get a good initial learning rate
        u = Π(z -  lr * F(z))
        lr = norm(z-u)/ norm(F(z)-F(u))
    end
    sumofsquares = 0
    cb(VI, z, lr, F(z), γ, 0.)
    get_anch_coeff = AnchCoeff(type=anchoring, VI=VI, normalizing_factor=norm(F(z0))^2)

    k = 0
    while k <= params.n_grad_eval
        # extrapolation step
        Fz = F(z)
        anch_coeff = get_anch_coeff(Fz, z)
        u = Π(z - lr*Fz + anch_coeff * (z0-z + lr*Fz))

        # update step
        Fu = F(u)
        z_new = Π(z - γ*lr*Fu + anch_coeff * (z0-z))

        # update stepsize
        if adaptive
            lr = min(lr, norm(u-z)/norm(Fu-Fz))
            if ada_heuristic
                lr = min(lr, norm(u-z_new)/norm(Fu-F(z_new)))
            end
        end
        sumofsquares += norm(Fz-Fu)^2
        if universal; lr = 1/sqrt(1+sumofsquares) end

        z = z_new
        k+=2
        cb(VI, u, lr, Fu, γ, k)
    end
end


function ogda(VI::AbstractVI, params::ProblemParams, cb::Callback; γ::Float64=1., adaptive=false)
    """Optimistic GDA for VI given by F with option for adaptive step size"""

    F, Π = VI.F, VI.Π
    z = z0 = params.z0

    if γ == 1. # regular OGDA
        lr = 1/(2*params.L)
    else # OGDA+
        lr = (1-γ)/((1+γ)*params.L)
    end

    F_old = F(z)

    if adaptive
        z_new = Π(z - lr*F(z))
        lr = 1/2 * norm(z_new - z)/ norm(F(z_new)-F(z))
    else
        F_new = F_old
    end
    cb(VI, z0, lr, F_old, γ, 0)

    k = 0
    while k <= params.n_grad_eval
        # TODO: with nonconstant stepsize this should be modified
        z_new = Π(z - lr*((1+γ)*F_new - F_old))

        if adaptive; lr = min(lr, 1/2 * norm(z_new - z)/norm(F_new - F_old)) end
        F_old, z = F_new, z_new
        F_new = F(z)

        k+=1
        cb(VI, z_new, lr, F_new, γ, k)
    end
end


function golden_ratio(VI::AbstractVI, params::ProblemParams, cb::Callback; ϕ::Float64=0., γ=missing, adaptive=false, anchoring=missing)
    """Yura's (adaptive) golden ratio method for VI. Stepsize is allowed to increase from iteration to the next.
    """

    # set default values
    if ϕ == 0.; adaptive ? ϕ = 1.5 : ϕ = (1+sqrt(5))/2 end
    if ismissing(γ) γ = 1/ϕ + 1/ϕ^2 end

    F, Π = VI.F, VI.Π
    z0 = z = params.z0
    lr = ϕ/(2*params.L)
    # lr = (2-ϕ)/(params.L)

    Fz = F(z)
    get_anch_coeff = AnchCoeff(type=anchoring, VI=VI, normalizing_factor=norm(Fz)^2)

    # compute initial stepsize
    if adaptive
        Fz_old, z_old = F(z), z
        z = Π(z - lr*Fz_old)
        # paper suggests to use for z a pertubation of z0
        Fz = F(z)
        lr = norm(z-z0)/norm(Fz-F(z0))
        lr_old = lr_old_old = lr
    end
    zbar = z

    cb(VI, z0, lr, F(z0), ϕ-1, 0)

    k = 0
    while k <= params.n_grad_eval

        if adaptive
            lr = min(γ * lr_old, ϕ^2/(4*lr_old_old) * norm(z-z_old)^2/ norm(Fz - Fz_old)^2)
            # lr = min(ρ * lr_old, 1/lr_old_old * norm(z-z_old)^2/ norm(Fz - Fz_old)^2)
            lr_old_old, lr_old = lr_old, lr
            z_old, Fz_old = z, Fz
        end

        zbar = (ϕ-1)/ϕ * z + 1/ϕ * zbar

        anch_coeff = get_anch_coeff(Fz, z)
        z = Π(zbar - lr*Fz + anch_coeff * (z0-zbar))

        Fz = F(z)
        k+=1
        cb(VI, z, lr, Fz, ϕ-1, k)
    end
end


function adaptive_EG(VI::AbstractVI, params::ProblemParams, cb::Callback; ν=0.99, τ=0.9, backtrack=true, count_grad_eval=true, α=missing)
    """Extra-gradient method from the paper 'escaping' limit cycles. Designed
    for weak Minty problems.  Adaptively computes a reduction in the update step
    compared to the extrapolation step.
    """

    ρ = -params.ρ/2 # use their definition of ρ
    if ismissing(ρ); ρ = -Inf end
    F, Π = VI.F, VI.Π
    z, u = params.z0, params.z0
    lr = 1/params.L # generous starting value so it does not start too small
    Fu = F(u)

    # some dummy values so the first callback is ok
    α = 1/2

    k = 0
    while k <= params.n_grad_eval

        # update stepsize
        if backtrack; lr, eval_bt = backtracking_ls(F, Π, z, lr, ν, τ, true) end
        cb(VI, u, lr, Fu, α, k)

        # update iterates
        Fz = F(z)
        u = Π(z - lr*Fz)
        Fu = F(u)
        Hu = u - lr*Fu
        Hz = z - lr*Fz

        # if theoretical bound (lr > -2ρ) not satisfied use smallest acceptable
        tmp = max(-0.499*lr, ρ)
        α = tmp/lr + dot(u-z, Hu-Hz) / norm(Hu-Hz)^2
        # α = clamp(α, 0.0001, 1)

        # TODO: maybe 2*α is better?
        z = z + α*(Hu - Hz)

        k+= 2
        if count_grad_eval k+= eval_bt end

    end
    cb(VI, u, lr, Fu, α, k)
end

function curvature_ls(F, z, ν=0.99)
    J(u) = ForwardDiff.jacobian(F, u)
    norm_hessian(u) = opnorm(J(u))
    if any(isnan.(z))
        return 0
        # TODO: should actually stop computation
    else
        lr_init = ν/norm_hessian(z)
    end

    return lr_init
end


function backtracking_ls(F, Π, z, lr::Float64, ν=0.9, τ=0.9, curvature=true)

    k = 0  # count extra gradient evals due to backtracking
    if curvature
        lr = curvature_ls(F, z)
    else # try to increase first

        increase = false
        Fz = F(z)
        Gz = Π(z-lr*Fz)
        # if we start with feasible stepsize, increase to maximum
        while lr * norm(F(Gz) -Fz) <= (ν * norm(Gz - z)) && k < 20
            increase = true
            lr = lr * 1/τ
            Gz = Π(z-lr*Fz)
            k += 1
        end
        # if we increased, stop. no need to do bt in the other direction
        if increase; return lr*τ, k end
    end

    Fz = F(z)
    Gz = Π(z-lr*Fz)
    # if stepsize is not feasible, decrease until it is
    while true
        if lr * norm(F(Gz) -Fz) <= (ν * norm(Gz - z)) || k > 20
            return lr, k
        end
        lr = τ * lr
        Gz = Π(z-lr*Fz)
        k += 1
    end
end


Base.@kwdef mutable struct AnchCoeff
    type
    VI::AbstractVI
    normalizing_factor::Float64
    G = 0.
    index = 1
end

function (o::AnchCoeff)(Fz, z)
    """I've been trying different ways of anchoring. This function unifies this across all methods """

    if ismissing(o.type)
        anch_coeff = 0
    elseif o.type == "normal"
        anch_coeff = 1/(o.index+1)
    elseif o.type == "acc"
        anch_coeff = 1/(o.index+10)
    elseif o.type == "adaptive"
        fixed_point_residual = norm(z - o.VI.Π(z - Fz))^2
        o.G += 1/fixed_point_residual
        anch_coeff = minimum([1/o.index, 1/o.normalizing_factor * 1/o.G])
        # if (0 == k % 100); println(anch_coeff) end
    end

    o.index += 1

    return anch_coeff
end
