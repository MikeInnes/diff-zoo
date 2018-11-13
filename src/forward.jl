# Implementing Forward Mode
# =========================
#
# In the [intro notebook](./intro.ipynb) we covered forward-mode differentiation
# thoroughly, but we don't yet have a real implementation that can work on our
# programs. Implementing AD effectively and efficiently is a field of its own,
# and we'll need to learn a few more tricks to get off the ground.

include("utils.jl");

# Up to now, we have differentiated things by creating a new Wengert list which
# contains parts of the original expression.

y = Wengert(:(5sin(log(x))))
derive(y, :x)

# We're now going to explicitly split our lists into two pieces: the original
# expression, and a new one which only calculates derivatives (but might refer
# back to values from the first). For example:

y = Wengert(:(5sin(log(x))))
#-
dy = derive(y, :x, out = Wengert(variable = :dy))
#-
Expr(dy)

# If we want to distinguish them, we can call `y` the *primal code* and `dy`
# the *tangent code*. Nothing fundamental has changed here, but it's useful
# to organise things this way.
#
# Almost all of the subtlety in differentiating programs comes from a
# mathematically trivial question: in what order do we evaluate the statements
# of the Wengert list? We have discussed the
# [forward/reverse](./backandforth.ipynb) distinction, but even once that choice
# is made, we have plenty of flexibility, and those choices can affect efficiency.
#
# For example, imagine if we straightforwardly evaluate `y` followed by `dy`. If
# we only cared about the final output of `y`, this would be no problem at all,
# but in general `dy` also needs to re-use variables like `y1` (or possibly any
# $y_i$). If our primal Wengert list has, say, a billion instructions, we end up
# having to store a billion intermediate $y_i$ before we run our tangent code.
#
# Alternatively, one can imagine running each instruction of the tangent code as
# early as possible; as soon as we run `y1 = log(x)`, for example, we know we
# can run `dy2 = cos(y1)` also. Then our final, combined program would look
# something like this:

# ```julia
# y0 = x
# dy = 1
# y1 = log(y0)
# dy = dy/y0
# y2 = cos(y1)
# dy = dy*sin(y1)
#   ...
# ```

# Now we can throw out `y1` soon after creating it, and we no longer have to
# store those billion intermediate results.
#
# The ability to do this is a very general property of forward differentiation;
# once we run $a = f(b)$, we can then run $\frac{da}{dx} = \frac{da}{db}
# \frac{db}{dx}$ using only `a` and `b`. It's really just a case of replacing
# basic instructions like `cos` with versions that calculate both the primal and
# tangent at once.

# Dual Numbers
# ------------
#
# Finally, the trick that we've been building up to: making our programming
# language do this all for us! Almost all common languages – with the notable
# exception of C – provide good support for *operator overloading*, which allows
# us to do exactly this replacement.
#
# To start with, we'll make a container that holds both a $y$ and a
# $\frac{dy}{dx}$, called a *dual number*.

struct Dual{T<:Real} <: Real
  x::T
  ϵ::T
end

Dual(1, 2)
#-
Dual(1.0,2.0)

# Let's print it nicely.

Base.show(io::IO, d::Dual) = print(io, d.x, " + ", d.ϵ, "ϵ")

Dual(1, 2)

# And add some of our rules for differentiation. The rules have the same basic
# pattern-matching structure as the ones we originally applied to our Wengert
# list, just with different notation.

import Base: +, -, *, /
a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x * b.x, b.x * a.ϵ - a.x * b.ϵ)

Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))

Dual(2, 2) * Dual(3, 4)

# Finally, we'll hook into Julia's number promotion system; this isn't essential
# to understand, but just makes it easier to work with Duals since we can now
# freely mix them with other number types.

Base.convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.x), convert(T, x.ϵ))
Base.convert(::Type{Dual{T}}, x::Real) where T = Dual(convert(T, x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}

Dual(1, 2) * 3

# We already have enough to start taking derivatives of some simple functions.
# If we pass a dual number into a function, the $\epsilon$ component represents
# the derivative.

f(x) = x / (1 + x*x)

f(Dual(5., 1.))

# We can make a utility which allows us to differentiate any function.

D(f, x) = f(Dual(x, one(x))).ϵ

D(f, 5.)

# Dual numbers seem pretty scarcely related to all the Wengert list stuff we
# were talking about earlier. But we need take a closer look at how this is
# working. To start with, look at Julia's internal representation of `f`.

@code_typed f(1.0)

# This is just a Wengert list! Though the naming is a little different – `mul_float`
# rather than the more general `*` and so on – it's still essentially the same
# data structure we were working with earlier. Moreover, you'll recognise
# the code for the derivative, too!

@code_typed D(f, 1.0)

# This code is again the same as the Wengert list derivative we worked out at
# the very beginning of this handbook. The order of operations is just a little
# different, and there's the odd missing or new instruction due to the different
# set of optimisations that Julia applies. Still, we have not escaped our fundamental
# symbolic differentiation algorithm, just tricked the compiler into doing most
# of the work for us.

derive(Wengert(:(sin(cos(x)))), :x)
#-
@code_typed D(x -> sin(cos(x)), 0.5)

# What of data structures, control flow, function calls? Although these things
# are all present in Julia's internal "Wengert list", they end up being the same
# in the tangent program as in the primal; so an operator overloading approach
# need not deal with them explicitly to do the right thing. This won't be true
# when we come to talk more about reverse mode, which demands a more complex
# approach.

# Perturbation Confusion
# ----------------------
#
# Actually, that's not quite true. Operator-overloading-based forward mode
# *almost always* does the right thing, but it is not flawless. This more
# advanced section will talk about nested differentiation and the nasty bug that
# can come with it.
#
# We can differentiate any function we want, as long as we have the right
# primitive definitions for it. For example, the derivative of $\sin(x)$ is
# $\cos(x)$.

D(sin, 0.5), cos(0.5)

# We can also differentiate the differentiation operator itself. We'll find that
# the second derivative of $\sin(x)$ is $-\sin(x)$.

D(x -> D(sin, x), 0.5), -sin(0.5)

# This worked because we ended up nesting dual numbers. If we create a dual number
# whose $\epsilon$ component is another dual number, then we end up tracking the
# derivative of the derivative.

# The issue comes about when we close over a variable that *is itself* being
# differentiated.

D(x -> x*D(y -> x+y, 1), 1) # == 1

# The derivative $\frac{d}{dy} (x + y) = 1$, so this is equivalent to
# $\frac{d}{dx}x$, which should also be $1$. So where did this go wrong? The
# problem is that when we closed over $x$, we didn't just get a numeric value
# but a dual number with $\epsilon = 1$. When we then calculated $x + y$, both
# epsilons were added as if $\frac{dx}{dy} = 1$ (effectively $x = y$). If we had
# written this down, the answer would be correct.

D(x -> x*D(y -> y+y, 1), 1)

# I leave this second example as an excercise to the reader. Needless to say,
# this has caught out many an AD implementor.

D(x -> x*D(y -> x*y, 1), 4) # == 8

# More on Dual Numbers
# --------------------
#
# The above discussion presented dual numbers as essentially being a trick for
# applying the chain rule. I wanted to take the opportunity to present an
# alternative viewpoint, which might be appealing if, like me, you have any
# training in physics.
#
# Complex arithmetic involves a new number, $i$, which behaves like no other:
# specifically, because $i^2 = -1$. We'll introduce a number called $\epsilon$,
# which is a bit like $i$ except that $\epsilon^2 = 0$; this is effectively a
# way of saying the $\epsilon$ is a very small number. The relevance of this
# comes from the original definition of differentiation, which also requires
# $\epsilon$ to be very small.
#
# $$
# \frac{d}{dx} f(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon)-f(x)}{\epsilon}
# $$
#
# We can see how our definition of $\epsilon$ works out by applying it to
# $f(x+\epsilon)$; let's say that $f(x) = sin(x^2)$.
#
# \begin{align}
# f(x + \epsilon) &= \sin((x + \epsilon)^2) \\
#                 &= \sin(x^2 + 2x\epsilon + \epsilon^2) \\
#                 &= \sin(x^2 + 2x\epsilon) \\
#                 &= \sin(x^2)\cos(2x\epsilon) + \cos(x^2)\sin(2x\epsilon) \\
#                 &= \sin(x^2) + 2x\cos(x^2)\epsilon \\
# \end{align}
#
# A few things have happened here. Firstly, we directly expand $(x+\epsilon)^2$
# and remove the $\epsilon^2$ term. We expand $sin(a+b)$ and then apply a *small
# angle approximation*: for small $\theta$, $\sin(\theta) \approx \theta$ and
# $\cos(\theta) \approx 1$. (This sounds pretty hand-wavy, but does follow from
# our original definition of $\epsilon$ if we look at the Taylor expansion of
# both functions). Finally we can plug this into our derivative rule.
#
# \begin{align}
# \frac{d}{dx} f(x) &= \frac{f(x+\epsilon)-f(x)}{\epsilon} \\
#                   &= 2x\cos(x^2)
# \end{align}
#
# This is, in my opinion, a rather nice way to derive functions by hand.
#
# This also leads to another nice trick, and a third way to look at forward-mode
# AD; if we replace $x + \epsilon$ with $x + \epsilon i$ then we still have
# $(\epsilon i)^2 = 0$. If $\epsilon$ is a small real number (say
# $1\times10^{-10}$), this is still true within floating point error, so our
# derivative still works out.

ϵ = 1e-10im
x = 0.5

f(x) = sin(x^2)

(f(x+ϵ) - f(x)) / ϵ
#-
2x*cos(x^2)

# So complex numbers can be used to get exact derivatives! This is very efficient
# and can be written using only one call to `f`.

imag(f(x+ϵ)) / imag(ϵ)

# Another way of looking at this is that we are doing bog-standard numerical
# differentiation, but the use of complex numbers avoids the typical problem
# with that technique (i.e. that a small perturbation ends up being overwhelmed
# by floating point error). The dual number is then a slight variation which
# makes the limit $\epsilon \rightarrow 0$ exact, rather than approximate.
# Forward mode AD can be described as "just" a clever implementation of
# numerical differentiation.
