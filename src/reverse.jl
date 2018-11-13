include("utils.jl");

# Source to Source Reverse Mode
# =============================
#
# [Forward mode](./forward.ipynb) works well because all of the symbolic
# operations happen at Julia's compile time; Julia can then optimise the
# resulting program (say, by applying SIMD instructions) and we get very fast
# derivative code. Although we can differentiate Julia code by [compiling it to
# a Wengert list](./tracing.ipynb), we'd be much better off if we could handle
# Julia code directly; then reverse mode can benefit from these optimisations
# too.
#
# However, Julia code is much more complex than a Wengert list, with constructs
# like control flow, data structures and function calls. To do this we'll have
# to handle each of these things in turn.
#
# The first thing to realise is that Julia code is much closer to a Wengert list
# than it looks. Despite its rich syntax, the compiler works with a Wengert-like
# format. The analyses and optimisations that compilers already carry out also
# benefit from this easily-work-with structure.

f(x) = x / (1 + x^2)

@code_typed f(1.0)

# Code with control flow is pnly a little different. We add `goto` statements
# and a construct called the "phi function"; the result is called [SSA
# form](https://en.wikipedia.org/wiki/Static_single_assignment_form).

function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

pow(2, 3)

@code_typed pow(2, 3)

# The details of this format are not too important. SSA form is powerful but
# somewhat fiddly to work with in practice, so the aim of this notebook is
# to give a broad intuition for how we handle this.
