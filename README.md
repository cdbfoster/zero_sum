zero_sum
====

An analysis engine for zero-sum games.

This crate provides a number of traits that can be used to facilitate the
implementation of a zero-sum game, and to allow the analysis thereof.

[Documentation](https://cdbfoster.github.io/doc/zero_sum)

# Usage

This crate is [on crates.io](https://crates.io/crates/zero_sum) and can be
used by adding `zero_sum` to the dependencies in your project's `Cargo.toml`.

```toml
[dependencies]
zero_sum = "0.2"
```

and add this to your crate root:

```rust
#[macro_use]
extern crate zero_sum;
```

# Implementation

The three basic traits are `Ply`, `Resolution`, and `State`.  These form
the basic building blocks of any zero-sum game.

In order to provide analysis, one must also create an evaluation type
(usually a tuple wrapper around a numeric type, i.e. `struct Eval(i32);`)
with `analysis::Evaluation`, and implement `analysis::Evaluatable` and
`analysis::Extrapolatable` on the `State` type.

# Example

A working example can be found in [examples/tic_tac_toe.rs](https://github.com/cdbfoster/zero_sum/blob/master/examples/tic_tac_toe.rs).
