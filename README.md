# lad_rs

**Logical Analysis of Data (LAD) in Rust**

`lad_rs` is a Rust implementation of Logical Analysis of Data (LAD) for binary classification. It provides a pipeline for converting tabular data into a binary representation, generating discriminative LAD patterns, and using those patterns for classification.

The library is built around two main components:

* `Binarizer` — converts numerical, nominal, and Boolean features into binary features.
* `RuleGenerator` — searches for LAD patterns and uses the resulting rules to make predictions.

The implementation uses [Polars](https://pola.rs/) for dataframe operations and [Rayon](https://github.com/rayon-rs/rayon) for parallel pattern evaluation.

## Pipeline

```text
                    Training Data
                         │
                         ▼
                  ┌──────────────┐
                  │  Binarizer   │
                  └──────┬───────┘
                         │
             Binary Feature Representation
                         │
                         ▼
                 ┌──────────────┐
                 │ RuleGenerator│
                 └──────┬───────┘
                        │
                 LAD Pattern Search
                        │
                        ▼
                  Prime Patterns
                        │
                        ▼
              ┌────────────────────┐
              │    Prediction      │
              └────────────────────┘
                 │       │       │
               exact   fuzzy   fallback
```

## How it works

### 1. Feature Binarization

LAD operates on binary variables, so `Binarizer` converts the input dataframe into Boolean features.

Different feature types are handled differently.

#### Boolean features

Boolean features are represented directly as a binary condition.

For example:

```text
is_tcp
```

becomes a condition such as:

```text
is_tcp = true
```

#### Nominal features

Categorical features are assigned binary codes.

For example, a feature with four distinct values:

```text
{TCP, UDP, ICMP, OTHER}
```

can be represented using two binary columns:

```text
feature_bit_0
feature_bit_1
```

The number of bits is determined by:

```text
ceil(log₂(number of unique values))
```

#### Numerical features

Numerical features are converted into intervals using recursively generated cut points.

A numerical feature might produce cut points such as:

```text
         c₁       c₂       c₃
─────────┬────────┬────────┬──────────►
         │        │        │
```

These divide the feature into ordered classes:

```text
x ≤ c₁
c₁ < x ≤ c₂
c₂ < x ≤ c₃
x > c₃
```

The resulting interval classes are then binary encoded.

### Cut-point selection

For numerical features, candidate cut points are evaluated using a class-separation score.

For each candidate split, the implementation calculates the proportion of each class on one side of the split and measures how different those proportions are.

The score is:

```text
                         1/2
        ┌──────────────────────────────┐
        │ Σ (rᵢ - rⱼ)²                 │
score = │ ───────────────────────────── │
        │          maximum              │
        └──────────────────────────────┘
```

where `rᵢ` is the proportion of class `i` in the current partition.

The `threshold` supplied to `Binarizer` determines the minimum score required for a cut point to be considered.

Cut points are generated recursively up to the configured `depth`.

---

## 2. LAD Pattern Generation

Once the data has been binarized, `RuleGenerator` searches for combinations of Boolean feature terms.

A pattern is represented internally as:

```rust
HashSet<(bool, usize)>
```

where each pair represents:

```text
(feature value, feature index)
```

For example:

```text
feature_0 = true
feature_3 = false
feature_7 = true
```

represents the logical pattern:

```text
feature_0 ∧ ¬feature_3 ∧ feature_7
```

### Pattern search

Pattern generation starts with individual feature terms and progressively constructs higher-degree patterns.

Conceptually:

```text
Degree 1
    │
    ├── f₁
    ├── ¬f₁
    ├── f₂
    ├── ¬f₂
    └── ...
    │
    ▼
Degree 2
    │
    ├── f₁ ∧ f₂
    ├── f₁ ∧ ¬f₂
    ├── ¬f₁ ∧ f₃
    └── ...
    │
    ▼
Degree 3
    │
    └── ...
```

Patterns that cover no observations are discarded.

Patterns are also evaluated using the LAD separation score. Patterns below the configured binarization threshold are removed from consideration.

### Prime patterns

When a pattern covers observations belonging to only one class, it can be selected as a classification rule.

The covered observations are removed from the corresponding class group, allowing subsequent patterns to cover previously uncovered observations.

This produces a collection of **prime patterns**:

```text
Class A  ← pattern 1
Class B  ← pattern 2
Class A  ← pattern 3
Class C  ← pattern 4
...
```

The search continues until the remaining observations are exhausted or the configured feature depth is reached.

---

## Prediction

`RuleGenerator` provides three prediction methods.

### `predict`

```rust
predict(&data)
```

Applies the generated rules in order.

For each observation, the first rule that completely covers it determines the prediction.

Observations not covered by any rule receive the fallback class.

The fallback class is initially the largest class among the training data.

### `predict_exact`

```rust
predict_exact(&data)
```

Works like `predict`, but observations that are not covered by any generated rule remain `Null` rather than receiving the fallback class.

This is useful when you want to distinguish:

```text
classified by LAD
```

from:

```text
not covered by any LAD rule
```

### `predict_fuzzy`

```rust
predict_fuzzy(&data)
```

Uses partial pattern coverage to resolve observations that are not completely covered by a rule.

For each rule, the fraction of its terms satisfied by an observation contributes to that rule's class confidence. The class with the greatest accumulated confidence is used when an observation has not already been classified by an exact rule.

This provides a less restrictive alternative to exact rule matching.

---

## Usage

### Creating a binarizer

```rust
use lad_rs::Binarizer;

let mut binarizer = Binarizer::new(
    0.25, // score threshold
    10,   // nominal feature size
    10,   // maximum cut points per column
    4,    // numerical cut-point depth
);
```

Generate cut points from training data:

```rust
binarizer.generate_cutpoints(&data, &labels)?;
```

Transform a dataframe:

```rust
let binary_data = binarizer.transform(&data)?;
```

The resulting dataframe contains Boolean columns representing the original features.

### Generating rules

```rust
use lad_rs::RuleGenerator;

let mut generator = RuleGenerator::new(
    &binarizer,
    0, // maximum number of features in a pattern; 0 = all
    1, // rule-generation depth
);

let remaining = generator.fit(&data, &labels)?;
```

The generated rules can be inspected with:

```rust
let rules = generator.get_rules();
```

### Prediction

Exact/fallback prediction:

```rust
let predictions = generator.predict(&test_data)?;
```

Exact-only prediction:

```rust
let predictions = generator.predict_exact(&test_data)?;
```

Fuzzy prediction:

```rust
let predictions = generator.predict_fuzzy(&test_data)?;
```

---

## Parameters

### `Binarizer`

```rust
Binarizer::new(
    threshold,
    nominal_size,
    max_cutpoints_per_column,
    depth,
)
```

| Parameter                  | Description                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `threshold`                | Minimum class-separation score accepted during cut-point/pattern evaluation |
| `nominal_size`             | Maximum number of unique values treated as nominal                          |
| `max_cutpoints_per_column` | Maximum cut points intended for a numerical feature                         |
| `depth`                    | Maximum recursion depth when generating numerical cut points                |

> **Implementation note:** `max_cutpoints_per_column` is currently stored by `Binarizer`, but the limiting operation is disabled in the current implementation. It therefore does not currently restrict the generated cut points.

### `RuleGenerator`

```rust
RuleGenerator::new(
    &binarizer,
    max,
    deep,
)
```

| Parameter   | Description                                                                     |
| ----------- | ------------------------------------------------------------------------------- |
| `binarizer` | Binarizer used to transform input data                                          |
| `max`       | Maximum number of features allowed in the pattern search; `0` uses all features |
| `deep`      | Controls repeated rule extraction from the same class during a search depth     |

---

## Data Flow

Training:

```text
DataFrame + Labels
        │
        ▼
generate_cutpoints()
        │
        ▼
Binarizer
        │
        ▼
Binary training data
        │
        ▼
RuleGenerator::fit()
        │
        ▼
Pattern generation
        │
        ▼
Pattern scoring
        │
        ▼
Prime patterns
        │
        ▼
Classification rules
```

Inference:

```text
New DataFrame
      │
      ▼
Binarizer::transform()
      │
      ▼
Binary data
      │
      ▼
Generated patterns
      │
      ▼
predict()
      │
      ▼
Predictions
```

---

## Parallelism

Pattern coverage across class-specific dataframes is evaluated in parallel using [Rayon](https://github.com/rayon-rs/rayon).

The core counting operation uses parallel iteration:

```text
                    Pattern
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Class 1      Class 2      Class 3
       coverage     coverage     coverage
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Counts
```

This allows class-level coverage calculations to execute concurrently.

---

## Dependencies

| Crate                                               | Purpose                                |
| --------------------------------------------------- | -------------------------------------- |
| [Polars](https://crates.io/crates/polars)           | DataFrame and columnar data processing |
| [Rayon](https://crates.io/crates/rayon)             | Parallel pattern evaluation            |
| [itertools](https://crates.io/crates/itertools)     | Iterator utilities                     |
| [notify-rust](https://crates.io/crates/notify-rust) | Desktop progress notifications         |

## Building

Clone the repository:

```bash
git clone https://github.com/Bha-Gu/lad_rs.git
cd lad_rs
```

Build in debug mode:

```bash
cargo build
```

Build an optimized release:

```bash
cargo build --release
```

The release profile uses aggressive link-time optimization:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
```

---

## Project Structure

```text
lad_rs/
├── src/
│   ├── lib.rs
│   └── binarization/
│       ├── binarize.rs
│       └── rule_generation.rs
├── Cargo.toml
└── README.md
```

The public library exports:

```rust
pub use crate::binarization::{
    binarize::Binarizer,
    rule_generation::RuleGenerator,
};
```

so the primary API can be accessed directly with:

```rust
use lad_rs::{Binarizer, RuleGenerator};
```

---

## Computational Considerations

LAD pattern generation is combinatorial. As the number of binary features and allowed pattern depth increase, the number of candidate patterns can grow rapidly.

For this reason, the following parameters have a significant effect on execution time:

* Number of input features
* Number of generated binary features
* Pattern depth
* Number of classes
* Pattern coverage
* Score threshold

Limiting the maximum pattern size and numerical cut-point depth can substantially reduce the search space.

The implementation therefore favors explicit bounds over unrestricted pattern generation.

---

## Current Status

`lad_rs` is a **research/experimental implementation** of LAD in Rust.

The current implementation focuses on:

* Tabular data binarization
* Numerical cut-point generation
* Binary encoding of nominal features
* LAD pattern generation
* Prime-rule extraction
* Exact rule-based prediction
* Fuzzy prediction
* Parallel pattern evaluation

The API and implementation are subject to change.

## Background

Logical Analysis of Data is a combinatorial approach to classification based on Boolean representations of observations and logical patterns.

This implementation was developed as part of work on **LAD-based intrusion detection**, including the IDS-LAD research project.

### Related publication

Bhaumikaditya Guleria, Maroti Deshmukh, and Rakhi Nautiyal.

**IDS-LAD: Intrusion detection system using logical analysis of data.**

*Cluster Computing*, 2025.

## License

See [LICENSE](LICENSE) for licensing information.
