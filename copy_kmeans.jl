# K-means algorithm

using Distances
using NearestNeighbors

using Printf
using LinearAlgebra
using SparseArrays
using Statistics

import Base: show
import StatsBase: IntegerVector, RealVector, RealMatrix, counts

include("utils.jl")

#### Interface

# C is the type of centers, an (abstract) matrix of size (d x k)
# D is the type of pairwise distance computation from points to cluster centers
# WC is the type of cluster weights, either Int (in the case where points are
# unweighted) or eltype(weights) (in the case where points are weighted).
"""
    KmeansResult{C,D<:Real,WC<:Real} <: ClusteringResultReal

The output of [`kmeans`](@ref) and [`kmeans!`](@ref).

# Type parameters
 * `C<:AbstractMatrix{<:AbstractFloat}`: type of the `centers` matrix
 * `D<:Real`: type of the assignment cost
 * `WC<:Real`: type of the cluster weight
"""
struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResultReal
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    wcounts::Vector{WC}        # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

wcounts(clu::KmeansResult) = clu.wcounts

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

"""
    kmeans!(X, centers; [kwargs...]) -> KmeansResult

Update the current cluster `centers` (``d×k`` matrix, where ``d`` is the
dimension and ``k`` the number of centroids) using the ``d×n`` data
matrix `X` (each column of `X` is a ``d``-dimensional data point).

See [`kmeans`](@ref) for the description of optional `kwargs`.
"""
function kmeans!(X::AbstractMatrix{<:Real},                # in: data matrix (d x n)
                 centers::AbstractMatrix{<:AbstractFloat}; # in: current centers (d x k)
                 weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: data point weights (n)
                 maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                 tol::Real=_kmeans_default_tol,            # in: tolerance of change at convergence
                 display::Symbol=_kmeans_default_display,  # in: level of display
                 distance::SemiMetric=SqEuclidean())       # in: function to compute distances
    d, n = size(X)
    dc, k = size(centers)
    WC = (weights === nothing) ? Int : eltype(weights)
    D = typeof(one(eltype(centers)) * one(WC))

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("k must be from 1:n (n=$n), k=$k given."))
    if weights !== nothing
      length(weights) == n || throw(DimensionMismatch("Incorrect length of weights."))
    end
    if k == n # each point in its own cluster
      return KmeansResult(copyto!(centers, X), collect(1:k), zeros(D, k), fill(1, k),
                          weights !== nothing ? copy(weights) : fill(1, k), D(0), 0, true)
    else
      if k == 1 # all points belong to the single cluster
        mean!(centers, X)
      end
      return _kmeans!(X, weights, centers, Int(maxiter), Float64(tol),
                      display_level(display), distance)
    end
end


"""
    kmeans(X, k, [...]) -> KmeansResult

K-means clustering of the ``d×n`` data matrix `X` (each column of `X`
is a ``d``-dimensional data point) into `k` clusters.

# Arguments
 - `init` (defaults to `:kmpp`): how cluster seeds should be initialized, could
   be one of the following:
   * a `Symbol`, the name of a seeding algorithm (see [Seeding](@ref) for a list
     of supported methods);
   * an instance of [`SeedingAlgorithm`](@ref);
   * an integer vector of length ``k`` that provides the indices of points to
     use as initial seeds.
 - `weights`: ``n``-element vector of point weights (the cluster centers are
   the weighted means of cluster members)
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)
"""
function real_kmeans(X::AbstractMatrix{<:Real},                # in: data matrix (d x n) columns = obs
                k::Integer;                               # in: number of centers
                weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: data point weights (n)
                init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=
                        _kmeans_default_init,             # in: initialization algorithm
                maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                tol::Real=_kmeans_default_tol,            # in: tolerance  of change at convergence
                display::Symbol=_kmeans_default_display,  # in: level of display
                distance::SemiMetric=SqEuclidean())       # in: function to calculate distance with
    d, n = size(X)
    (1 <= k <= n) || throw(ArgumentError("k must be from 1:n (n=$n), k=$k given."))

    # initialize the centers using a type wide enough so that the updates
    # centers[i, cj] += X[i, j] * wj will occur without loss of precision through rounding
    T = float(weights === nothing ? eltype(X) : promote_type(eltype(X), eltype(weights)))
    iseeds = initseeds(init, X, k)
    centers = copyseeds!(Matrix{T}(undef, d, k), X, iseeds)

    kmeans!(X, centers;
            weights=weights, maxiter=Int(maxiter), tol=Float64(tol),
            display=display, distance=distance)
end

#### Core implementation

# core k-means skeleton
function _kmeans!(X::AbstractMatrix{<:Real},                # in: data matrix (d x n)
                  weights::Union{Nothing, Vector{<:Real}},  # in: data point weights (n)
                  centers::AbstractMatrix{<:AbstractFloat}, # in/out: matrix of centers (d x k)
                  maxiter::Int,                             # in: maximum number of iterations
                  tol::Float64,                             # in: tolerance of change at convergence
                  displevel::Int,                           # in: the level of display
                  distance::SemiMetric)                     # in: function to calculate distance
    d, n = size(X)
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # whether a center needs to be updated
    unused = Vector{Int}()
    num_affected = k # number of centers to which dists need to be recomputed

    # assign containers for the vector of assignments & number of data points assigned to each cluster
    assignments = Vector{Int}(undef, n)
    counts = Vector{Int}(undef, k)

    # compute pairwise distances, preassign costs and cluster weights
    dmat = Distances.pairwise(distance, centers, X, dims=2)
    WC = (weights === nothing) ? Int : eltype(weights)
    wcounts = Vector{WC}(undef, k)
    D = typeof(one(eltype(dmat)) * one(WC))
    costs = Vector{D}(undef, n)

    update_assignments!(dmat, true, assignments, costs, counts,
                        to_update, unused)
    objv = weights === nothing ? sum(costs) : dot(weights, costs)

    # main loop
    t = 0
    converged = false
    if displevel >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
        @printf("%7d %18.6e\n", t, objv)
    end

    while !converged && t < maxiter
        t += 1

        # update (affected) centers
        update_centers!(X, weights, assignments, to_update, centers, wcounts)

        if !isempty(unused)
            repick_unused_centers(X, costs, centers, unused, distance)
            to_update[unused] .= true
        end

        if t == 1 || num_affected > 0.75 * k
            Distances.pairwise!(dmat, distance, centers, X, dims=2)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = findall(to_update)
            Distances.pairwise!(view(dmat, affected_inds, :), distance,
                      view(centers, :, affected_inds), X, dims=2)
        end

        # update assignments
        update_assignments!(dmat, false, assignments, costs, counts,
                            to_update, unused)
        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence
        prev_objv = objv
        objv = weights === nothing ? sum(costs) : dot(weights, costs)
        objv_change = objv - prev_objv

        if objv_change > tol
            @warn("The clustering cost increased at iteration #$t")
        elseif (k == 1) || (abs(objv_change) < tol)
            converged = true
        end

        # display information (if required)
        if displevel >= 2
            @printf("%7d %18.6e %18.6e | %8d\n", t, objv, objv_change, num_affected)
        end
    end

    if displevel >= 1
        if converged
            println("K-means converged with $t iterations (objv = $objv)")
        else
            println("K-means terminated without convergence after $t iterations (objv = $objv)")
        end
    end

    return KmeansResult(centers, assignments, costs, counts,
                        wcounts, objv, t, converged)
end

#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!(dmat::Matrix{<:Real},     # in:  distance matrix (k x n)
                             is_init::Bool,            # in:  whether it is the initial run
                             assignments::Vector{Int}, # out: assignment vector (n)
                             costs::Vector{<:Real},    # out: costs of the resultant assignment (n)
                             counts::Vector{Int},      # out: # of points assigned to each cluster (k)
                             to_update::Vector{Bool},  # out: whether a center needs update (k)
                             unused::Vector{Int}       # out: list of centers with no points assigned
                             )
    k, n = size(dmat)

    # re-initialize the counting vector
    fill!(counts, 0)

    if is_init
        fill!(to_update, true)
    else
        fill!(to_update, false)
        if !isempty(unused)
            empty!(unused)
        end
    end

    # process each point
    @inbounds for j = 1:n
        # find the closest cluster to the i-th point. Note that a
        # is necessarily between 1 and size(dmat, 1) === k as a result
        # and can thus be used as an index in an `inbounds` environment
        c, a = findmin(view(dmat, :, j))

        # set/update the assignment
        if is_init
            assignments[j] = a
        else  # update
            pa = assignments[j]
            if pa != a
                # if assignment changes,
                # both old and new centers need to be updated
                assignments[j] = a
                to_update[a] = true
                to_update[pa] = true
            end
        end

        # set costs and counts accordingly
        costs[j] = c
        counts[a] += 1
    end

    # look for centers that have no assigned points
    for i = 1:k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where points are not weighted)
#
function update_centers!(X::AbstractMatrix{<:Real},        # in: data matrix (d x n)
                         weights::Nothing,                 # in: point weights
                         assignments::Vector{Int},         # in: assignments (n)
                         to_update::Vector{Bool},          # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:AbstractFloat}, # out: updated centers (d x k)
                         wcounts::Vector{Int})             # out: updated cluster weights (k)
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    wcounts[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip points assigned to a center that doesn't need to be updated
        cj = assignments[j]
        if to_update[cj]
            if wcounts[cj] > 0
                for i in 1:d
                    centers[i, cj] += X[i, j]
                end
            else
                for i in 1:d
                    centers[i, cj] = X[i, j]
                end
            end
            wcounts[cj] += 1
        end
    end

    # sum ==> mean
    @inbounds for j in 1:k
        if to_update[j]
            cj = wcounts[j]
            for i in 1:d
                centers[i, j] /= cj
            end
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where points are weighted)
#
function update_centers!(X::AbstractMatrix{<:Real}, # in: data matrix (d x n)
                         weights::Vector{W},        # in: point weights (n)
                         assignments::Vector{Int},  # in: assignments (n)
                         to_update::Vector{Bool},   # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:Real}, # out: updated centers (d x k)
                         wcounts::Vector{W}         # out: updated cluster weights (k)
                         ) where W<:Real
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    wcounts[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip points with negative weights or assigned to a center
        # that doesn't need to be updated
        wj = weights[j]
        cj = assignments[j]
        if wj > 0 && to_update[cj]
            if wcounts[cj] > 0
                for i in 1:d
                    centers[i, cj] += X[i, j] * wj
                end
            else
                for i in 1:d
                    centers[i, cj] = X[i, j] * wj
                end
            end
            wcounts[cj] += wj
        end
    end

    # sum ==> mean
    @inbounds for j in 1:k
        if to_update[j]
            cj = wcounts[j]
            for i in 1:d
                centers[i, j] /= cj
            end
        end
    end
end


#
#  Re-picks centers that have no points assigned to them.
#
function repick_unused_centers(X::AbstractMatrix{<:Real}, # in: the data matrix (d x n)
                               costs::Vector{<:Real},     # in: the current assignment costs (n)
                               centers::AbstractMatrix{<:AbstractFloat}, # out: the centers (d x k)
                               unused::Vector{Int},       # in: indices of centers to be updated
                               distance::SemiMetric)      # in: function to calculate the distance with
    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(X, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = view(X, :, j)
        centers[:, i] = v
        colwise!(ds, distance, v, X)
        tcosts = min(tcosts, ds)
    end
end

# """
#     SeedingAlgorithm
# Base type for all seeding algorithms.
# Each seeding algorithm should implement the two functions: [`initseeds!`](@ref)
# and [`initseeds_by_costs!`](@ref).
# """
# abstract type SeedingAlgorithm end

# """
#     initseeds(alg::Union{SeedingAlgorithm, Symbol},
#               X::AbstractMatrix, k::Integer) -> Vector{Int}
# Select `k` seeds from a ``d×n`` data matrix `X` using the `alg`
# algorithm.
# `alg` could be either an instance of [`SeedingAlgorithm`](@ref) or a symbolic
# name of the algorithm.
# Returns the vector of `k` seed indices.
# """
# initseeds(alg::SeedingAlgorithm, X::AbstractMatrix{<:Real}, k::Integer; kwargs...) =
#     initseeds!(Vector{Int}(undef, k), alg, X; kwargs...)

# """
#     initseeds_by_costs(alg::Union{SeedingAlgorithm, Symbol},
#                        costs::AbstractMatrix, k::Integer) -> Vector{Int}
# Select `k` seeds from the ``n×n`` `costs` matrix using algorithm `alg`.
# Here, `costs[i, j]` is the cost of assigning points `i`` and ``j``
# to the same cluster. One may, for example, use the squared Euclidean distance
# between the points as the cost.
# Returns the vector of `k` seed indices.
# """
# initseeds_by_costs(alg::SeedingAlgorithm, costs::AbstractMatrix{<:Real}, k::Integer; kwargs...) =
#     initseeds_by_costs!(Vector{Int}(undef, k), alg, costs; kwargs...)

# seeding_algorithm(s::Symbol) =
#     s == :rand ? RandSeedAlg() :
#     s == :kmpp ? KmppAlg() :
#     s == :kmcen ? KmCentralityAlg() :
#     throw(ArgumentError("Unknown seeding algorithm $s"))

# function check_seeding_args(n::Integer, k::Integer)
#     k >= 1 || throw(ArgumentError("The number of seeds ($k) must be positive."))
#     k <= n || throw(ArgumentError("Cannot select more seeds ($k) than data points ($n)."))
# end

# check_seeding_args(X::AbstractMatrix, iseeds::AbstractVector) =
#     check_seeding_args(size(X, 2), length(iseeds))

# initseeds(algname::Symbol, X::AbstractMatrix{<:Real}, k::Integer; kwargs...) =
#     initseeds(seeding_algorithm(algname), X, k; kwargs...)::Vector{Int}

# initseeds_by_costs(algname::Symbol, costs::AbstractMatrix{<:Real}, k::Integer; kwargs...) =
#     initseeds_by_costs(seeding_algorithm(algname), costs, k; kwargs...)

# # use specified vector of seeds
# function initseeds(iseeds::AbstractVector{<:Integer}, X::AbstractMatrix{<:Real}, k::Integer; kwargs...)
#     length(iseeds) == k ||
#         throw(ArgumentError("The length of seeds vector ($(length(iseeds))) differs from the number of seeds requested ($k)"))
#     check_seeding_args(X, iseeds)
#     n = size(X, 2)
#     # check that seed indices are fine
#     for (i, seed) in enumerate(iseeds)
#         (1 <= seed <= n) || throw(ArgumentError("Seed #$i refers to an incorrect data point ($seed)"))
#     end
#     # NOTE no duplicate checks are done, should we?
#     convert(Vector{Int}, iseeds)
# end
# initseeds_by_costs(iseeds::AbstractVector{<:Integer}, costs::AbstractMatrix{<:Real}, k::Integer; kwargs...) =
#     initseeds(iseeds, costs, k; kwargs...) # NOTE: passing costs as X, but should be fine since only size(X, 2) is used

# function copyseeds!(S::AbstractMatrix{<:AbstractFloat},
#                     X::AbstractMatrix{<:Real},
#                     iseeds::AbstractVector)
#     d, n = size(X)
#     k = length(iseeds)
#     size(S) == (d, k) ||
#         throw(DimensionMismatch("Inconsistent seeds matrix dimensions: $((d, k)) expected, $(size(S)) given."))
#     return copyto!(S, view(X, :, iseeds))
# end

# """
#     RandSeedAlg <: SeedingAlgorithm
# Random seeding (`:rand`).
# Chooses an arbitrary subset of ``k`` data points as cluster seeds.
# """
# struct RandSeedAlg <: SeedingAlgorithm end

# """
#     initseeds!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
#                X::AbstractMatrix) -> iseeds
# Initialize `iseeds` with the indices of cluster seeds for the `X` data matrix
# using the `alg` seeding algorithm.
# """
# function initseeds!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real};
#                     rng::AbstractRNG=Random.GLOBAL_RNG)
#     check_seeding_args(X, iseeds)
#     sample!(rng, 1:size(X, 2), iseeds; replace=false)
# end

# """
#     initseeds_by_costs!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
#                         costs::AbstractMatrix) -> iseeds
# Initialize `iseeds` with the indices of cluster seeds for the `costs` matrix
# using the `alg` seeding algorithm.
# Here, `costs[i, j]` is the cost of assigning points ``i`` and ``j``
# to the same cluster. One may, for example, use the squared Euclidean distance
# between the points as the cost.
# """
# function initseeds_by_costs!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real}; rng::AbstractRNG=Random.GLOBAL_RNG)
#     check_seeding_args(X, iseeds)
#     sample!(rng, 1:size(X,2), iseeds; replace=false)
# end

# """
#     KmppAlg <: SeedingAlgorithm
# Kmeans++ seeding (`:kmpp`).
# Chooses the seeds sequentially. The probability of a point to be chosen is
# proportional to the minimum cost of assigning it to the existing seeds.
# # References
# > D. Arthur and S. Vassilvitskii (2007).
# > *k-means++: the advantages of careful seeding.*
# > 18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.
# """
# struct KmppAlg <: SeedingAlgorithm end

# function initseeds!(iseeds::IntegerVector, alg::KmppAlg,
#                     X::AbstractMatrix{<:Real},
#                     metric::PreMetric = SqEuclidean();
#                     rng::AbstractRNG=Random.GLOBAL_RNG)
#     n = size(X, 2)
#     k = length(iseeds)
#     check_seeding_args(n, k)

#     # randomly pick the first center
#     p = rand(rng, 1:n)
#     iseeds[1] = p

#     if k > 1
#         mincosts = colwise(metric, X, view(X, :, p))
#         mincosts[p] = 0

#         # pick remaining (with a chance proportional to mincosts)
#         tmpcosts = zeros(n)
#         for j = 2:k
#             p = wsample(rng, 1:n, mincosts)
#             iseeds[j] = p

#             # update mincosts
#             c = view(X, :, p)
#             colwise!(tmpcosts, metric, X, view(X, :, p))
#             updatemin!(mincosts, tmpcosts)
#             mincosts[p] = 0
#         end
#     end

#     return iseeds
# end

# function initseeds_by_costs!(iseeds::IntegerVector, alg::KmppAlg,
#                              costs::AbstractMatrix{<:Real};
#                              rng::AbstractRNG=Random.GLOBAL_RNG)
#     n = size(costs, 1)
#     k = length(iseeds)
#     check_seeding_args(n, k)

#     # randomly pick the first center
#     p = rand(rng, 1:n)
#     iseeds[1] = p

#     if k > 1
#         mincosts = costs[:, p]
#         mincosts[p] = 0

#         # pick remaining (with a chance proportional to mincosts)
#         for j = 2:k
#             p = wsample(rng, 1:n, mincosts)
#             iseeds[j] = p

#             # update mincosts
#             updatemin!(mincosts, view(costs, :, p))
#             mincosts[p] = 0
#         end
#     end

#     return iseeds
# end

# """
#     KmCentralityAlg <: SeedingAlgorithm
# K-medoids initialization based on centrality (`:kmcen`).
# Choose the ``k`` points with the highest *centrality* as seeds.
# # References
# > Hae-Sang Park and Chi-Hyuck Jun.
# > *A simple and fast algorithm for K-medoids clustering.*
# > doi:10.1016/j.eswa.2008.01.039
# """
# struct KmCentralityAlg <: SeedingAlgorithm end

# function initseeds_by_costs!(iseeds::IntegerVector, alg::KmCentralityAlg,
#                              costs::AbstractMatrix{<:Real}; kwargs...)

#     n = size(costs, 1)
#     k = length(iseeds)
#     check_seeding_args(n, k)

#     # compute score for each item
#     coefs = vec(sum(costs, dims=2))
#     for i = 1:n
#         @inbounds coefs[i] = inv(coefs[i])
#     end

#     # scores[j] = \sum_j costs[i,j] / (\sum_{j'} costs[i,j'])
#     #           = costs[i,j] * coefs[i]
#     scores = costs'coefs

#     # lower score indicates better seeds
#     sp = sortperm(scores)
#     for i = 1:k
#         @inbounds iseeds[i] = sp[i]
#     end
#     return iseeds
# end

# initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::AbstractMatrix{<:Real},
#            metric::PreMetric = SqEuclidean(); kwargs...) =
#     initseeds_by_costs!(iseeds, alg, pairwise(metric, X, dims=2); kwargs...)