# Forward- and Reverse-Mode Differentiation
# =========================================

include("utils.jl");

# Differentiation tools are frequently described as implementing "forward mode"
# or "reverse mode" AD. This distinction was briefly covered in the intro
# notebook, but here we'll go into more detail. We'll start with an intuition
# for what the distinction *means* in terms of the differentiation process; then
# we'll discuss why it's an important consideration in practice.

# Consider a simple mathematical expression:

y = :(sin(x^2) * 5)

# Written as a Wengert list:

Wengert(y)

# The ability to take derivatives mechanically relies on two things: Firstly, we
# know derivatives for each basic function in our program (e.g.
# $\frac{dy_2}{dy_1}=cos(y_1)$). Secondly, we have a rule of composition called the
# *chain rule* which lets us compose these basic derivatives together.
#
# $$
# \frac{dy}{dx} = \frac{dy_1}{dx} \times \frac{dy_2}{dy_1} \times \frac{dy}{dy_2}
# $$
#
# More specifically:
#
# $$
# \begin{align}
# \frac{dy}{dx} &= 2x \times cos(y_1) \times 5 \\
#               &= 2x \times cos(x^2) \times 5
# \end{align}
# $$
#
# The forward/reverse distinction basically amounts to: given that we do
# multiplications one at a time, do we evaluate $\frac{dy_1}{dx} \times
# \frac{dy_2}{dy_1}$ first, or $\frac{dy_2}{dy_1} \times \frac{dy}{dy_2}$? (This seems
# like a pointless question right now, given that either gets us the same
# results, but bear with me.)
#
# It's easier to see the distinction if we think algorithmically. Given some
# enormous Wengert list with $n$ instructions, we have two ways to differentiate
# it:
#
# **(1)**: Start with the known quantity $\frac{dy_0}{dx} = \frac{dx}{dx} = 1$
# at the beginning of the list. Look up the derivative for the next instruction
# $\frac{dy_{i+1}}{dy_i}$ and multiply out the top, getting $\frac{dy_1}{dx}$,
# $\frac{dy_2}{dx}$, ... $\frac{dy_{n-1}}{dx}$, $\frac{dy}{dx}$. Because we
# walked forward over the Wengert list, this is called *forward mode*. Each
# intermediate derivative $\frac{dy_i}{dx}$ is known as a *perturbation*.
#
# **(2)**: Start with the known quantity $\frac{dy}{dy_n} = \frac{dy}{dy} = 1$
# at the end of the list. Look up the derivative for the previous instruction
# $\frac{dy_i}{dy_{i-1}}$ and multiply out the bottom, getting
# $\frac{dy}{dy_n}$, $\frac{dy}{dy_{n-1}}$, ... $\frac{dy}{dy_1}$,
# $\frac{dy}{dx}$. Because we walked in reverse over the Wengert list, this is
# called *reverse mode*. Each intermediate derivative $\frac{dy}{dy_i}$ is known
# as a *sensitivity*.
#
# This all seems very academic, so we need to explain why it might make a
# difference to performance. I'll give two related explanations: dealing with
# mulitple variables, and working with vectors rather than scalars.

# Explanation 1: Multiple Variables
# ---------------------------------
#
# So far we have dealt only with simple functions that take a number, and return
# a number. But more generally we'll deal with functions that take, or produce,
# multiple numbers of interest.
#
# For example what if we have a function that returns *two* numbers, and we want
# derivatives for both? Do we have to do the whole thing twice over?

y = quote
  y2 = sin(x^2)
  y3 = y2 * 5
end

# Let's say we want both of the derivatives $\frac{dy_2}{dx}$ and
# $\frac{dy_3}{dx}$. You can probably see where this is going now; the Wengert
# list representation of this expression has not actually changed!

Wengert(y)

# Now, we discussed that when doing forward mode differentiation, we actually
# calculate *every* intermediate derivative $\frac{dy_i}{dx}$, which means we get
# $\frac{dy_2}{dx}$ for free. This property goes all the way back to our
# original, recursive formulation of differentiation, which calculated the
# derivatives of a complex expression by combining the derivatives of simpler
# ones.

derive(Wengert(y), :x)

# In our output, $y_7 = \frac{dy_2}{dx}$ and $y_8 = \frac{dy_3}{dx}$.
#
# Let's consider the opposite situation, a function of two variables $a$ and
# $b$, where we'd like to get $\frac{dy}{da}$ and $\frac{dy}{db}$.

y = :(sin(a) * b)
#-
Wengert(y)

# This one is a bit tougher. We can start the forward-mode differentiation
# process with $\frac{da}{da} = 1$ or with $\frac{db}{db} = 1$, but if we want
# both we'll have to go through the entire multiplying-out process twice.
#
# But both variables ultimately end up at the same place, $y$, and we know that
# $\frac{dy}{dy} = 1$. Aha, so perhaps we can use reverse mode for this
# instead!
#
# Exactly opposite to forward mode, reverse mode gives us every intermediate
# gradient $\frac{dy_i}{dy}$ for free, ultimately leading back in the inputs
# $\frac{da}{dy}$ and $\frac{db}{dy}$.
#
# It's easy to see, then, why reverse-mode differentiation – or backpropagation
# – is so effective for machine learning. In general we have a large computation
# with millions of parameters, yet only a single scalar loss to optimise. We can
# get gradients even for these millions of inputs in a single pass, enabling ML
# to scale to complex tasks like image and voice recognition.

# Explanation 2: Vector Calculus
# ------------------------------
#
# So far we have dealt only with simple functions that take a number, and return
# a number. But more generally we'll deal with functions that take, or produce,
# *vectors* containing multiple numbers of interest.
#
# It's useful to consider how our idea of differentiation works when we have
# vectors. For example, a function that takes a vector of length $2$ to another
# vector of length $2$:

f(x) = [x[1] * x[2], cos(x[1])]

x = [2, 3]
y = f(x)

# We now need to talk about what we mean by $\frac{d}{dx}f(x)$, given that we
# can't apply the usual limit rule. What we *can* do is take the derivative of
# any scalar *element* of $y$ with respect to any element of $x$. (We'll use
# subscripts $x_n$ to refer to the $n^{th}$ index of $x$.) For example:
#
# $$
# \begin{align}
# \frac{dy_1}{dx_1} &= \frac{d}{dx_1} x_1 \times x_2 = x_2 \\
# \frac{dy_1}{dx_2} &= \frac{d}{dx_2} x_1 \times x_2 = x_1 \\
# \frac{dy_2}{dx_1} &= \frac{d}{dx_1} \cos(x_1) = -\sin(x_1) \\
# \frac{dy_2}{dx_2} &= \frac{d}{dx_2} \cos(x_1) = 0 \\
# \end{align}
# $$
#
# It's a little easier if we organise all of these derivatives into a matrix.
#
# $$
# J_{ij} = \frac{dy_i}{dx_j}
# $$
#
# This $2\times2$ matrix is called the *Jacobian*, and in general it's what we mean by
# $\frac{dy}{dx}$. (The Jacobian for a scalar function like $y = \sin(x)$ only
# has one element, so it's consistent with our current idea of the derivative
# $\frac{dy}{dx}$.) The key point here is that the Jacobian is a potentially
# large object: it has a size `length(y) * length(x)`. Now, we discussed that
# the distinction between forward and reverse mode is whether we propagate
# $\frac{dy_i}{dx}$ or $\frac{dy}{dy_i}$, which can have a size of either
# `length(y_i) * length(x)` or `length(y) * length(y_i)`.
#
# It should be clear, then, what mode is better if we have a gazillion inputs
# and one output. In forward mode we need to carry around a gazillion
# "perturbations" *for each* element of $y_i$, whereas in reverse we only need a
# gradient of the same size of $x$. And vice versa.
