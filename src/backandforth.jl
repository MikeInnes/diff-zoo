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
# $\frac{db}{da}=cos(a)$). Secondly, we have a rule of composition called the
# *chain rule* which lets us compose these basic derivatives together.
#
# $$
# \frac{dy}{dx} = \frac{da}{dx} \times \frac{db}{da} \times \frac{dy}{db}
# $$
#
# More specifically:
#
# $$
# \begin{align}
# \frac{dy}{dx} &= 2x \times cos(a) \times 5 \\
#               &= 2x \times cos(x^2) \times 5
# \end{align}
# $$
#
# The forward/reverse distinction basically amounts to: given that we do
# multiplications one at a time, do we evaluate $\frac{da}{dx} \times
# \frac{db}{da}$ first, or $\frac{db}{da} \times \frac{dy}{db}$? (This seems
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

# Explanation 2: Vector Calculus
# ------------------------------
