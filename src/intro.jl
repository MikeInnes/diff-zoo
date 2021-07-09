# Differentiation for Hackers
# ===========================
#
# These notebooks are an exploration of various approaches to analytical
# differentiation. Differentiation is something you learned in school; we start
# with an expression like $y = 3x^2 + 2x + 1$ and find an expression for the
# derivative like $\frac{dy}{dx} = 6x + 2$. Once we have such an expression, we
# can *evaluate* it by plugging in a specific value for $x$ (say 0.5) to find
# the derivative at that point (in this case $\frac{dy}{dx} = 5$).
#
# Despite its surface simplicity, this technique lies at the core of all modern
# machine learning and deep learning, alongside many other parts of statistics,
# mathematical optimisation and engineering. There has recently been an
# explosion in automatic differentiation (AD) tools, all with different designs
# and tradeoffs, and it can be difficult to understand how they relate to each
# other.
#
# We aim to fix this by beginning with the "calculus 101" rules that you are
# familiar with and implementing simple symbolic differentiators over mathematical
# expressions. Then we show how tweaks to this basic framework generalise from
# expressions to programming languages, leading us to modern automatic
# differentiation tools and machine learning frameworks like TensorFlow and
# PyTorch, and giving us a unified view across the AD landscape.

# Symbolic Differentiation
# ------------------------
#
# To talk about derivatives, we need to talk about *expressions*, which are
# symbolic forms like $x^2 + 1$ (as opposed to numbers like $5$). Normal Julia
# programs only work with numbers; we can write down $x^2 + 1$ but this only
# lets us calculate its value for a specific $x$.

x = 2
y = x^2 + 1

# However, Julia also offers a *quotation operator* which lets us talk about the
# expression itself, without needing to know what $x$ is.

y = :(x^2 + 1)
#-
typeof(y)

# Expressions are a tree data structure. They have a `head` which tells us what
# kind of expression they are (say, a function call or if statement). They have
# `args`, their children, which may be further sub-expressions. For example,
# $x^2 + 1$ is a call to $+$, and one of its children is the expression $x^2$.

y.head
#-
y.args

# We could have built this expression by hand rather than using quotation. It's
# just a bog-standard tree data structure that happens to have nice printing.

x2 = Expr(:call, :^, :x, 2)
#-
y = Expr(:call, :+, x2, 1)

# We can evaluate our expression to get a number out.

eval(y)

# When we differentiate something, we'll start by manipulating an expression
# like this, and then we can optionally evaluate it with numbers to get a
# numerical derivative. I'll call these the "symbolic phase" and the "numeric
# phase" of differentiation, respectively.

# How might we differentiate an expression like $x^2 + 1$? We can start by
# looking at the basic rules in differential calculus.
#
# $$
# \begin{align}
# \frac{d}{dx} x &= 1 \\
# \frac{d}{dx} (-u) &= - \frac{du}{dx} \\
# \frac{d}{dx} (u + v) &= \frac{du}{dx} + \frac{dv}{dx} \\
# \frac{d}{dx} (u * v) &= v \frac{du}{dx} + u \frac{dv}{dx} \\
# \frac{d}{dx} (u / v) &= (v \frac{du}{dx} - u \frac{dv}{dx}) / v^2 \\
# \frac{d}{dx} u^n &= n u^{n-1} \\
# \end{align}
# $$
#
# Seeing $\frac{d}{dx}(u)$ as a function, these rules look a lot like a
# recursive algorithm. To differentiate something like `y = a + b`, we
# differentiate `a` and `b` and combine them together. To differentiate `a` we
# do the same thing, and so on; eventually we'll hit something like `x` or `3`
# which has a trivial derivative ($1$ or $0$).

# Let's start by handling the obvious cases, $y = x$ and $y = 1$.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  error("$ex is not differentiable")
end
#-
y = :(x)
derive(y, :x)
#-
y = :(1)
derive(y, :x)

# We can look for expressions of the form `y = a + b` using pattern matching,
# with a package called
# [MacroTools](https://github.com/MikeInnes/MacroTools.jl). If `@capture`
# returns true, then we can work with the sub-expressions `a` and `b`.

using MacroTools

y = :(x + 1)
#-
@capture(y, a_ * b_)
#-
@capture(y, a_ + b_)
#-
a, b

# Let's use this to add a rule to `derive`, following the chain rule above.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
  error("$ex is not differentiable")
end
#-
y = :(x + 1)
derive(y, :x)
#-
y = :(x + (1 + (x + 1)))
derive(y, :x)

# These are the correct derivatives, even if they could be simplified a bit. We
# can go on to add the rest of the rules similarly.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
  @capture(ex, a_ * b_) ? :($a * $(derive(b, x)) + $b * $(derive(a, x))) :
  @capture(ex, a_^n_Number) ? :($(derive(a, x)) * ($n * $a^$(n-1))) :
  @capture(ex, a_ / b_) ? :(($b * $(derive(a, x)) - $a * $(derive(b, x))) / $b^2) :
  error("$ex is not differentiable")
end

# This is enough to get us a slightly more difficult derivative.

y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# This is correct – it's equivalent to $6x + 2$ – but it's also a bit noisy, with a
# lot of redundant terms like $x + 0$. We can clean this up by creating some
# smarter functions to do our symbolic addition and multiplication. They'll just
# avoid actually doing anything if the input is redundant.

addm(a, b) = a == 0 ? b : b == 0 ? a : :($a + $b)
mulm(a, b) = 0 in (a, b) ? 0 : a == 1 ? b : b == 1 ? a : :($a * $b)
mulm(a, b, c...) = mulm(mulm(a, b), c...)
powm(a, b) = b == 0 ? 1 : b == 1 ? a : :($a ^ $b)

#-
addm(:a, :b)
#-
addm(:a, 0)
#-
mulm(:b, 1)
#-
powm(:a, 1)

# Our tweaked `derive` function:

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? addm(derive(a, x), derive(b, x)) :
  @capture(ex, a_ * b_) ? addm(mulm(a, derive(b, x)), mulm(b, derive(a, x))) :
  @capture(ex, a_^n_Number) ? mulm(derive(a, x), n, powm(a, n-1)) :
  @capture(ex, a_ / b_) ? :(($(mulm(b, derive(a, x))) - $(mulm(a, derive(b, x)))) / $(powm(b, 2))) :
  error("$ex is not differentiable")
end

# And the output is much cleaner.

y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# Having done this, we can also calculate a nested derivative
# $\frac{d^2y}{dx^2}$, and so on.

ddy = derive(dy, :x)
#-
derive(ddy, :x)

# There is a deeper problem with our differentiation algorithm, though. Look at
# how big this derivative is.

derive(:(x / (1 + x^2)), :x)

# Adding an extra `* x` makes it even bigger! There's a bunch of redundant work
# here, repeating the expression $1 + x^2$ three times over.

derive(:(x / (1 + x^2) * x), :x)

# This happens because our rules look like,
# $\frac{d(u*v)}{dx} = u*\frac{dv}{dx} + v*\frac{du}{dx}$.
# Every multiplication repeats the whole sub-expression and its derivative,
# making the output exponentially large in the size of its input.
#
# This seems to be an achilles heel for our little differentiator, since it will
# make it impractical to run on any realistically-sized program. But wait!
# Things are not quite as simple as they seem, because this expression is not
# *actually* as big as it looks.
#
# Imagine we write down:

y1 = :(1 * 2)
y2 = :($y1 + $y1 + $y1 + $y1)

# This looks like a large expression, but in actual fact it does not contain
# $1*2$ four times over, just four pointers to $y1$; it is not really a tree but
# a graph that gets printed as a tree. We can show this by explicitly printing
# the expression in a way that preserves structure.
#
# (The definition of `printstructure` is not important to understand, but is
# here for reference.)

printstructure(x, _, _) = x

function printstructure(ex::Expr, cache = IdDict(), n = Ref(0))
  haskey(cache, ex) && return cache[ex]
  args = map(x -> printstructure(x, cache, n), ex.args)
  cache[ex] = sym = Symbol(:y, n[] += 1)
  println(:($sym = $(Expr(ex.head, args...))))
  return sym
end

printstructure(y2);

# Note that this is *not* the same as running common subexpression elimination
# to simplify the tree, which would have an $O(n^2)$ computational cost. If
# there is real duplication in the expression, it'll show up
# (technically, that is because `IdDict` hashes by object-id and each `Expr`
# has different identity and object-id accordingly).

:(1*2 + 1*2) |> printstructure;

# This is effectively a change in notation: we were previously using a kind of
# "calculator notation" in which any computation used more than once had to be
# repeated in full. Now we are allowed to use variable bindings to get the same
# effect.

# If we try `printstructure` on our differentiated code, we'll see that the
# output is not so bad after all:

:(x / (1 + x^2)) |> printstructure;
#-
derive(:(x / (1 + x^2)), :x)
#-
derive(:(x / (1 + x^2)), :x) |> printstructure;

# The expression $x^2 + 1$ is now defined once and reused rather than being
# repeated, and adding the extra `* x` now adds a couple of instructions to our
# derivative, rather than doubling its size. It turns out that our "naive"
# symbolic differentiator actually preserves structure in a very sensible way,
# and we just needed the right program representation to exploit that.

derive(:(x / (1 + x^2) * x), :x)
#-
derive(:(x / (1 + x^2) * x), :x) |> printstructure;

# Calculator notation – expressions without variable bindings – is a terrible
# format for anything, and will tend to blow up in size whether you
# differentiate it or not. Symbolic differentiation is commonly criticised for
# its susceptibility to "expression swell", but in fact has nothing to do with
# the differentiation algorithm itself, and we need not change it to get better
# results.
#
# Conversely, the way we have used `Expr` objects to represent variable bindings
# is perfectly sound, if a little unusual. This format could happily be used to
# illustrate all of the concepts in this handbook, and the recursive algorithms
# used to do so are elegant. However, it will clarify some things if they are
# written a little more explicitly; for this we'll introduce a new, equivalent
# representation for expressions.

# The Wengert List
# ----------------
#
# The output of `printstructure` above is known as a "Wengert List", an explicit
# list of instructions that's a bit like writing assembly code. Really, Wengert
# lists are nothing more or less than mathematical expressions written out
# verbosely, and we can easily convert to and from equivalent `Expr` objects.

include("utils.jl");
#-
y = :(3x^2 + (2x + 1))
#-
wy = Wengert(y)
#-
Expr(wy)

# Inside, we can see that it really is just a list of function calls, where
# $y_n$ refers to the result of the $n^{th}$.

wy.instructions

# Like `Expr`s, we can also build Wengert lists by hand.

w = Wengert()
tmp = push!(w, :(x^2))
w
#-
push!(w, :($tmp + 1))
w

# Armed with this, we can quite straightforwardly port our recursive symbolic
# differentiation algorithm to the Wengert list.

function derive(ex, x, w)
  ex isa Variable && (ex = w[ex])
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? push!(w, addm(derive(a, x, w), derive(b, x, w))) :
  @capture(ex, a_ * b_) ? push!(w, addm(mulm(a, derive(b, x, w)), mulm(b, derive(a, x, w)))) :
  @capture(ex, a_^n_Number) ? push!(w, mulm(derive(a, x, w), n, powm(a, n-1))) :
  @capture(ex, a_ / b_) ? push!(w, :(($(mulm(b, derive(a, x, w))) - $(mulm(a, derive(b, x, w)))) / $(powm(b, 2)))) :
  error("$ex is not differentiable")
end

derive(w::Wengert, x) = (derive(w[end], x, w); w)

# It behaves identically to what we wrote before; we have only changed the
# underlying representation.

derive(Wengert(:(3x^2 + (2x + 1))), :x) |> Expr

# In fact, we can compare them directly using the `printstructure` function we
# wrote earlier.

derive(:(x / (1 + x^2)), :x) |> printstructure;
#-
derive(Wengert(:(x / (1 + x^2))), :x)

# They are *almost* identical; the only difference is the unused variable `y3`
# in the Wengert version. This happens because our `Expr` format effectively
# removes dead code for us automatically. We'll see the same thing happen if
# we convert the Wengert list back into an `Expr`.

derive(Wengert(:(x / (1 + x^2))), :x) |> Expr
#-

function derive(w::Wengert, x)
  ds = Dict()
  ds[x] = 1
  d(x) = get(ds, x, 0)
  for v in keys(w)
    ex = w[v]
    Δ = @capture(ex, a_ + b_) ? addm(d(a), d(b)) :
        @capture(ex, a_ * b_) ? addm(mulm(a, d(b)), mulm(b, d(a))) :
        @capture(ex, a_^n_Number) ? mulm(d(a), n, powm(a,n-1)) :
        @capture(ex, a_ / b_) ? :(($(mulm(b, d(a))) - $(mulm(a, d(b)))) / $(powm(b, 2))) :
        error("$ex is not differentiable")
    ds[v] = push!(w, Δ)
  end
  return w
end

derive(Wengert(:(x / (1 + x^2))), :x) |> Expr

# One more thing. The astute reader may notice that our differentiation
# algorithm begins with $\frac{dx}{dx}=1$ and propagates this forward to the
# output; in other words it does [forward-mode
# differentiation](./backandforth.ipynb). We can tweak our code a little to do
# reverse mode instead.

function derive_r(w::Wengert, x)
  ds = Dict()
  d(x) = get(ds, x, 0)
  d(x, Δ) = ds[x] = haskey(ds, x) ? addm(ds[x],Δ) : Δ
  d(lastindex(w), 1)
  for v in reverse(collect(keys(w)))
    ex = w[v]
    Δ = d(v)
    if @capture(ex, a_ + b_)
      d(a, Δ)
      d(b, Δ)
    elseif @capture(ex, a_ * b_)
      d(a, push!(w, mulm(Δ, b)))
      d(b, push!(w, mulm(Δ, a)))
    elseif @capture(ex, a_^n_Number)
      d(a, mulm(Δ, n, :($(powm(a, n-1)))))
    elseif @capture(ex, a_ / b_)
      d(a, push!(w, :($(mulm(Δ, b)) / $(powm(b, 2)))))
      d(b, push!(w, :(-$(mulm(Δ, a)) / $(powm(b, 2)))))
    else
      error("$ex is not differentiable")
    end
  end
  push!(w, d(x))
  return w
end

# There are only two distinct algorithms in this handbook, and this is the
# second! It's quite similar to forward mode, with the difference that we
# walk backwards over the list, and each time we see a usage of a variable
# $y_i$ we accumulate a gradient for that variable.

derive_r(Wengert(:(x / (1 + x^2))), :x) |> Expr

# For now, the output looks pretty similar to that of forward mode; we'll
# explain why the [distinction makes a difference](./backandforth.ipynb) in future
# notebooks.

# Lastly, let's assert the differentiators we wrote so far are all correct.
y = :(x / (1 + x^2))

x = 0.5
dy = (1-x^2) / (1+x^2)^2 # hand-written derivative
@assert @show(derive(y, :x) |> eval) == dy
@assert @show(derive(Wengert(y), :x) |> Expr |> eval) == dy
@assert @show(derive_r(Wengert(y), :x) |> Expr |> eval) ≈ dy
