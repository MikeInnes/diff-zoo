# Differentiation for Hackers

[![Build Status](https://travis-ci.org/MikeInnes/diff-zoo.svg?branch=master)](https://travis-ci.org/MikeInnes/diff-zoo)

The goal of this handbook is to demystify *algorithmic differentiation*, the
tool that underlies modern machine learning. It begins with a calculus-101 style
understanding and gradually extends this to build toy implementations of systems
similar to PyTorch and TensorFlow. I have tried to clarify the relationships
between every kind of differentiation I can think of – including forward and
reverse, symbolic, numeric, tracing and source transformation. Where typical real-word ADs are mired in implementation details, these implementations are designed to be coherent enough that the real, fundamental differences – of which there are surprisingly few – become obvious.

The intro notebook is recommended to start with, but otherwise notebooks do not have a fixed order.

* [Intro](https://github.com/MikeInnes/diff-zoo/blob/notebooks/intro.ipynb) – explains the basics, beginning with a simple symbolic differentiation routine.
* [Back & Forth](https://github.com/MikeInnes/diff-zoo/blob/notebooks/backandforth.ipynb) – discusses the difference between forward and reverse mode AD.
* [Forward](https://github.com/MikeInnes/diff-zoo/blob/notebooks/forward.ipynb) – discusses forward-mode AD and its relationship to symbolic and numerical differentiation.
* [Tracing](https://github.com/MikeInnes/diff-zoo/blob/notebooks/tracing.ipynb) – discusses tracing-based implementations of reverse mode, as used by TensorFlow and PyTorch.
* [Reverse](https://github.com/MikeInnes/diff-zoo/blob/notebooks/reverse.ipynb) – discusses a more powerful reverse mode based on source transformation (not complete).

If you want to run the notebooks locally, they can be built by running the
`src/notebooks.jl` script using Julia. They should appear inside a `/notebooks`
folder. Alternatively, you can run through the scripts in Juno.
