zero_sum
====

An analysis engine for zero-sum games.

This crate provides a number of traits that can be used to facilitate the
implementation of a zero-sum game, and to allow the analysis thereof.

Also provided through the use of optional features are implementations
for tic-tac-toe and the game of [tak](http://cheapass.com/tak/).

# Usage

This crate is [on crates.io](https://crates.io/crates/zero_sum) and can be
used by adding `zero_sum` to the dependencies in your project's `Cargo.toml`.

```toml
[dependencies]
zero_sum = "1.0"
```

and add this to your crate root:

```rust
extern crate zero_sum;
# fn main() { }
```

If you want to implement the library, you'll need to include a `#[macro_use]`
line before `extern crate zero_sum;`

If you want to use one of the implementations provided inside the `zero_sum::impls`
module, you'll need to specify the appropriate features in your project's `Cargo.toml`:

```toml
[features]
default = [ "zero_sum/with_tak" ]
```

for instance, to include the `tak` module.

# Implementation

The three basic traits are `Ply`, `Resolution`, and `State`.  These form
the basic building blocks of any zero-sum game.

In order to provide analysis, one must also create an evaluation type
(usually a tuple wrapper around a numeric type, i.e. `struct Eval(i32);`)
with `analysis::Evaluation`, and implement `analysis::Evaluatable` and
`analysis::Extrapolatable` on the `State` type.

# Example

The provided tic-tac-toe implementation is very simple and a usage example can
be found in [examples/tic_tac_toe.rs](https://github.com/cdbfoster/zero_sum/blob/master/examples/tic_tac_toe.rs).
