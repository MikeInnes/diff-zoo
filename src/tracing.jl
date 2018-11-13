# Tracing-based Automatic Differentiation
# =======================================
#
# Machine learning primarily needs [reverse-mode AD](./backandforth.ipynb), and
# tracing / operator overloading approaches are by far the most popular way to
# it; this is the technique used by ML frameworks from Theano to PyTorch. This
# notebook will cover the techniques used by those frameworks, as well as
# clarifying the distinction between the "static declaration"
# (Theano/TensorFlow) and "eager execution" (Chainer/PyTorch/Flux) approaches to
# AD.

include("utils.jl")

# Partial Evaluation
# ------------------
#
# Say we have a simple implementation of $x^n$ which we want to differentiate.

function pow(x, n)
  r = 1
  for i = 1:n
    r *= x
  end
  return r
end

pow(2, 3)

# We already know how to [differentiate Wengert lists](./intro.ipynb), but this
# doesn't look much like one of those. In fact, we can't write this program as a
# Wengert list at all, given that it contains control flow; and more generally
# our programs might have things like data structures or function calls that we
# don't know how to differentiate either.
#
# Though it's possible to generalise the Wengert list to handle these things,
# there's actually a simple and surprisingly effective alternative, called
# "partial evaluation". This means running some part of a program without
# running all of it. For example, given an expression like $x + 5 * n$ where we
# know $n = 3$, we can simplify to $x + 15$ even though we don't know what $x$
# is. This is a common trick in compilers, and Julia will often do it for you:

f(x, n) = x + 5 * n
g(x) = f(x, 3)

code_typed(g, Tuple{Int})[1]

# This suggests a solution to our dilemma above. If we know what $n$ is (say,
# $3$), we can write `pow(x, 3)` as $((1*x)*x)*x$, which _is_ a Wengert
# expression that we can differentiate. In effect, this is a kind of compilation
# from a complex language (Julia, Python) to a much simpler one.

# Static Declaration
# ------------------
#
# We want to trace all of the basic mathematical operations in the program,
# stripping away everything else. We'll do this using Julia's operator
# overloading; the idea is to create a new type which, rather than actually executing
# operations like $a + b$, records them into a Wengert list.

import Base: +, -

struct Staged
  w::Wengert
  var
end

a::Staged + b::Staged = Staged(w, push!(a.w, :($(a.var) + $(b.var))))

a::Staged - b::Staged = Staged(w, push!(a.w, :($(a.var) - $(b.var))))

# Actually, all of our staged definitions follow the same pattern, so we can
# just do them in a loop. We also add an extra method so that we can multiply
# staged values by constants.

for f in [:+, :*, :-, :^, :/]
  @eval Base.$f(a::Staged, b::Staged) = Staged(a.w, push!(a.w, Expr(:call, $(Expr(:quote, f)), a.var, b.var)))
  @eval Base.$f(a, b::Staged) = Staged(b.w, push!(b.w, Expr(:call, $(Expr(:quote, f)), a, b.var)))
end

# The idea here is to begin by creating a Wengert list (the "graph" in ML
# framework parlance), and create some symbolic variables which do not yet
# have numerical values.

w = Wengert()
x = Staged(w, :x)
y = Staged(w, :y)

# When we manipulate these variables, we'll get Wengert lists.

z = 2x + y
z.w |> Expr

# Crucially, this works with our original `pow` function!

w = Wengert()
x = Staged(w, :x)

y = pow(x, 3)
y.w |> Expr

# The rest is almost too easy! We already know how to derive this.

dy = derive_r(y.w, :x)
Expr(dy)

# If we dump the derived code into a function, we get code for the derivative
# of $x^3$ at any point (i.e. $3x^2$).

@eval dcube(x) = $(Expr(dy))

dcube(5)

# Congratulations, you just implemented TensorFlow.

# Eager Execution
# ---------------

# This approach has a crucial problem; because it works by stripping out control
# flow and parameters like $n$, it effectively freezes all of these things. We
# can get a specific derivative for $x^3$, $x^4$ and so on, but we can't get the
# general derivative of $x^n$ with a single Wengert list. This puts a severe
# limitation on the kinds of models we can express.$^1$
#
# The solution? Well, just re-build the Wengert list from scratch every time!

function D(f, x)
  x_ = Staged(w, :x)
  dy = derive(f(x_).w, :x)
  eval(:(let x = $x; $(Expr(dy)) end))
end

D(x -> pow(x, 3), 5)
#-
D(x -> pow(x, 5), 5)

# This gets us our gradients, but it's not going to be fast – there's a lot of overhead
# to building and evaluating the list/graph every time. There are two things we can
# do to alleviate this:
#
# 1. Interpret, rather compile, the Wengert list.
# 2. Fuse interpretation of the list (the numeric phase) with the building
#    and manipulation of the Wengert list (the symbolic phase).
#
# Implementing this looks a lot like the `Staged` object above. The key difference
# is that alongside the Wengert list, we store the numerical values of each variable
# and instruction as we go along. Also, rather than explicitly naming variables
# `x`, `y` etc, we generate names using `gensym()`.

gensym()
#-
struct Tape
  instructions::Wengert
  values
end

Tape() = Tape(Wengert(), Dict())

struct Tracked
  w::Tape
  var
end

function track(t::Tape, x)
  var = gensym()
  t.values[var] = x
  Tracked(t, var)
end

Base.getindex(x::Tracked) = x.w.values[x.var]

for f in [:+, :*, :-, :^, :/]
  @eval function Base.$f(a::Tracked, b::Tracked)
    var = push!(a.w.instructions, Expr(:call, $(Expr(:quote, f)), a.var, b.var))
    a.w.values[var] = $f(a[], b[])
    Tracked(a.w, var)
  end
  @eval function Base.$f(a, b::Tracked)
    var = push!(b.w.instructions, Expr(:call, $(Expr(:quote, f)), a, b.var))
    b.w.values[var] = $f(a, b[])
    Tracked(b.w, var)
  end
  @eval function Base.$f(a::Tracked, b)
    var = push!(a.w.instructions, Expr(:call, $(Expr(:quote, f)), a.var, b))
    a.w.values[var] = $f(a[], b)
    Tracked(a.w, var)
  end
end

# Now, when we call `pow` it looks a lot more like we are dealing with normal
# numeric values; but there is still a Wengert list inside.

t = Tape()
x = track(t, 5)

y = pow(x, 3)
y[]

y.w.instructions |> Expr

# Finally, we need to alter how we derive this list. The key insight is that
# since we already have values available, we don't need to explicitly build
# and evaluate the derivative code; instead, we can just evaluate each instruction
# numerically as we go along. We more-or-less just need to replace our symbolic
# functions like (`addm`) with the regular ones (`+`).
#
# This is, of course, not a particularly optimised implementation, and faster
# versions have many more tricks up their sleaves. But this gets at all the key
# ideas.

function derive(w::Tape, xs...)
  ds = Dict()
  val(x) = get(w.values, x, x)
  d(x) = get(ds, x, 0)
  d(x, Δ) = ds[x] = d(x) + Δ
  d(lastindex(w.instructions), 1)
  for v in reverse(collect(keys(w.instructions)))
    ex = w.instructions[v]
    Δ = d(v)
    if @capture(ex, a_ + b_)
      d(a, Δ)
      d(b, Δ)
    elseif @capture(ex, a_ * b_)
      d(a, Δ * val(b))
      d(b, Δ * val(a))
    elseif @capture(ex, a_^n_Number)
      d(a, Δ * n * val(a) ^ (n-1))
    elseif @capture(ex, a_ / b_)
      d(a, Δ * val(b))
      d(b, -Δ*val(a)/val(b)^2)
    else
      error("$ex is not differentiable")
    end
  end
  return map(x -> d(x.var), xs)
end

derive(y.w, x)

# With this we can implement a more general gradient function.

function gradient(f, xs...)
  t = Tape()
  xs = map(x -> track(t, x), xs)
  f(xs...)
  derive(t, xs...)
end

# Even with the limited set of gradients that we have, we're well on our way to
# differentiating more complex programs, like a custom `sin` function.

gradient((a, b) -> a*b, 2, 3)
#-
mysin(x) = sum((-1)^k/factorial(1.0+2k) * x^(1+2k) for k = 0:5)
#-
gradient(mysin, 0.5)
#-
cos(0.5)

# We can even take nested derivatives!

gradient(x -> gradient(mysin, x)[1], 0.5)
#-
-sin(0.5)

# Though the tracing approach has significant limitations, its power is in how
# easy it is to implement: one can build a fairly full-featured implementation,
# in almost any language, in a weekend. Almost all languages have the
# operator-overloading features required, and no matter how complex the host
# language, one ends up with a simple Wengert list.

# Note that we have not removed the need to apply our basic symbolic
# differentiation algorithm here. We are still looking up gradient definitions,
# reversing data flow and applying the chain rule – it's just interleaved with
# our numerical operations, and we avoid building the output into an explicit
# Wengert list.
#
# It's somewhat unusual to emphasise the symbolic side of AD, but I think it
# gives us an incisive way to understand the tradeoffs that different systems
# make. For example: TensorFlow-style AD has its numeric phase separate from
# Python's runtime, which makes it awkward to use. Conversely, PyTorch does run
# its numerical phase at runtime, but also its symbolic phase, making it
# impossible to optimise the backwards pass.
#
# We [observed](./forward.ipynb) that OO-based forward mode is particularly
# successful because it carries out its symbolic and numeric operations at
# Julia's compile and run time, respectively. In the [source to source reverse
# mode](./reverse.ipynb) notebook, we'll explore doing this for reverse mode as
# well.

# ### Footnotes

# $^1$ Systems like TensorFlow can also just provide ways to inject control flow
# into the graph. This brings us closer to a [source-to-source
# approach](./reverse.ipynb) where Python is used to build an expression in
# TensorFlows internal graph language.

# Fun fact: PyTorch and Flux's tapes are actually closer to the `Expr` format
# that we originally used, in which "tracked" tensors just have pointers to
# their parents (implicitly forming a graph/Wengert list/expression tree). A
# naive algorithm for backpropagation suffers from exponential runtime for the
# *exact* same reason that naive symbolic diff does; "flattening" this graph
# into a tree causes it to blow up in size.
