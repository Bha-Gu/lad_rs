# lad_rs

**Logical Analysis of Data (LAD) implemented in Rust.**

`lad_rs` is a Rust implementation of **Logical Analysis of Data (LAD)** for classification of tabular datasets.

The library provides two main components:

* **`Binarizer`** — converts numerical, nominal, and Boolean features into binary representations and determines cut points for numerical features.
* **`RuleGenerator`** — searches the resulting binary feature space for class-discriminating logical patterns and uses those patterns as classification rules.

The implementation uses [Polars](https://pola.rs/) for tabular data processing and [Rayon](https://github.com/rayon-rs/rayon) for parallel pattern evaluation.

## Overview

LAD represents observations using Boolean variables and classifies them using logical patterns.

`lad_rs` follows this general pipeline:

```text
                 Input Data
                     │
                     ▼
              ┌─────────────┐
              │  Binarizer  │
              └──────┬──────┘
                     │
                     ▼
             Binary Data
                     │
                     ▼
             ┌─────────────┐
             │    LAD      │
             │    Rule     │
             │  Generation │
             └──────┬──────┘
                    │
                    ▼
              Prime Patterns
                    │
                    ▼
               Prediction
```

The implementation is designed primarily for experimentation and research rather than as a general-purpose machine-learning framework.

---

## Features

* Numerical feature binarization using recursively generated cut points
* Binary encoding of nominal features
* Boolean feature handling
* Configurable cut-point search depth
* Configurable numerical/nominal boundary
* Configurable maximum number of cut points per column
* LAD pattern generation
* Prime pattern extraction
* Class-specific pattern coverage
* Parallel pattern evaluation using Rayon
* Exact rule-based prediction
* Fuzzy prediction for uncovered samples
* Rule-index output for inspecting which rule classified a sample
* Model persistence to JSON and CSV files
* Multiple rule-generation implementations:

  * `fit`
  * `fit_new`
  * `fit_legacy`

---

# Binarization

LAD operates on binary variables. `Binarizer` converts the input dataframe into a collection of Boolean features.

```rust
use lad_rs::Binarizer;

let mut binarizer = Binarizer::new(
    0.25, // minimum split score
    10,   // numeric-as-nominal threshold
    20,   // maximum cut points per column
    4,    // numerical search depth
);
```

The binarizer first determines how each input feature should be represented.

## Feature types

### Boolean features

Boolean columns are represented directly.

A column such as:

```text
is_tcp
```

produces a Boolean feature corresponding to one of its values.

### Nominal features

Columns with a small number of unique values, as determined by `numeric_as_nominal_upper_threshold`, are treated as nominal.

String columns are also treated as nominal.

For example:

```text
protocol
────────
TCP
UDP
ICMP
```

is converted to binary columns using a compact binary encoding.

For `n` unique values, the number of generated bits is:

```text
ceil(log₂(n))
```

A value is assigned an integer index and the index is represented using binary bits.

### Numerical features

Numerical columns with sufficiently many unique values are processed using LAD-inspired cut-point generation.

The column is first sorted together with its class labels.

Candidate split points are evaluated according to how strongly the classes are separated by the split.

A selected cut point is placed between two adjacent distinct numerical values:

```text
x₁  x₂  x₃  x₄  x₅
       ▲
       │
    cut point
```

The actual cut point is the mean of the two neighboring values.

The search then recursively processes the resulting segments.

---

# Cut-point scoring

The binarizer uses a class-separation score based on the difference between class-wise coverage proportions.

For each class:

```text
rᵢ = samples of class i before the cut
     ────────────────────────────────
     total samples of class i
```

The score is based on pairwise differences:

```text
              Σᵢ<ⱼ (rᵢ - rⱼ)²
score = √( ───────────────────── )
                    M
```

where `M` is the maximum possible value for the number of classes.

For two classes this reduces to the absolute difference between their coverage proportions.

A cut point is considered only when its score reaches the configured `threshold`.

The search is repeated recursively up to `search_depth`.

---

# Transforming data

Once cut points have been generated, the same `Binarizer` can transform another dataframe:

```rust
binarizer.generate_cutpoints(&train_data, &train_labels)?;

let binary_train = binarizer.transform(&train_data)?;
let binary_test = binarizer.transform(&test_data)?;
```

This ensures that the test data is transformed using the representation learned from the training data.

Generated columns follow the form:

```text
feature_bit_0
feature_bit_1
feature_bit_2
...
```

---

# Rule Generation

`RuleGenerator` searches the binary feature space for patterns that distinguish classes.

A pattern is internally represented as:

```rust
HashSet<(bool, usize)>
```

Each element represents:

```text
(Boolean value, feature index)
```

For example:

```text
(feature_0 = true)
(feature_3 = false)
(feature_7 = true)
```

corresponds conceptually to:

```text
feature_0 ∧ ¬feature_3 ∧ feature_7
```

A pattern covers an observation when all of its terms are satisfied.

---

## Class-specific search

During training, observations are divided according to their class labels.

For every candidate pattern, `RuleGenerator` determines how many observations it covers in each class.

For example:

```text
             Pattern
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
    Class A   Class B   Class C
       80        0        0
```

A pattern that covers observations from only one class is a candidate for a prime pattern.

Once a selected pattern is extracted, the observations it covers are removed from the remaining search space.

The process is repeated to obtain a collection of rules.

---

# Pattern search

The current `fit` implementation searches patterns incrementally by complexity.

A simplified view of the search is:

```text
                     Single terms
                          │
                          ▼
                 Candidate patterns
                          │
                          ▼
                   Larger patterns
                          │
                          ▼
              Class-specific coverage
                          │
                          ▼
                 Best discriminating
                     patterns
                          │
                          ▼
                  Prime patterns
```

Pattern construction is restricted so that feature indices are introduced in increasing order. This avoids generating equivalent patterns through different insertion orders.

The `max` parameter limits the maximum number of features considered in a pattern.

```rust
let mut generator = RuleGenerator::new(&binarizer, 4);
```

A value of `0` means that all available features may be considered.

---

# Prime patterns

A selected pattern is stored together with:

1. The class it represents
2. The logical pattern
3. Its coverage ratio within that class
4. The number of samples it covers

Internally:

```rust
(usize, Pattern, f64, usize)
```

The public `get_rules()` method exposes the rule in a more convenient form:

```rust
let rules = generator.get_rules();
```

Each rule contains:

```text
(class, [(polarity, feature)], sample_count)
```

This makes it possible to inspect the generated LAD rules directly.

---

# Prediction

The generated rules can be applied to new data using several prediction modes.

## Standard prediction

```rust
let predictions = generator.predict(&test_data)?;
```

Each sample is assigned to the first rule that completely covers it.

If no rule covers a sample, the fallback class is used.

The fallback class is initially the largest class remaining in the training data.

---

## Exact prediction

```rust
let (predictions, rule_indices, binary_data) =
    generator.predict_exact(&test_data, until)?;
```

`predict_exact` only considers rules whose stored sample count is at least `until`.

It returns three values:

```text
Predictions
RuleIndex
Transformed Data
```

`RuleIndex` identifies the rule that classified each sample.

Samples not covered by the selected rules remain `Null`.

This makes `predict_exact` useful when the goal is to examine the direct coverage of LAD rules rather than force a prediction for every sample.

---

## Fuzzy prediction

```rust
let (predictions, rule_indices, binary_data) =
    generator.predict_fuzzy(&test_data, until)?;
```

Fuzzy prediction first applies exact rule coverage.

For samples that remain unclassified, the implementation constructs class-specific feature statistics from the selected rules.

For every class and feature, it tracks the weighted occurrence of:

```text
feature = true
feature = false
```

These values are normalized into class-specific proportions.

For an uncovered sample, its Boolean feature values are then compared against these proportions to produce a similarity value for each class.

The class with the greatest similarity is selected.

Thus fuzzy prediction provides a way to classify observations that are not completely covered by an individual LAD rule.

---

# Model persistence

Both `Binarizer` and `RuleGenerator` support saving and loading their state.

## Binarizer

```rust
binarizer.save(
    "binarizer.json",
    "binarizer_data",
)?;
```

The configuration is stored in JSON while the generated cut-point series are stored as CSV files.

The model can subsequently be restored:

```rust
let binarizer =
    Binarizer::load(
        "binarizer.json",
        "binarizer_data",
    )?;
```

The resulting directory contains files conceptually like:

```text
binarizer.json
cutpoints_0.csv
cutpoints_1.csv
cutpoints_2.csv
...
```

---

## RuleGenerator

A trained rule generator can also be persisted:

```rust
generator.save(
    "model.json",
    "model_data",
)?;
```

The saved representation contains:

```text
model.json
model_data/
├── binarizer.json
├── labels.csv
├── cutpoints_0.csv
├── cutpoints_1.csv
└── ...
```

It can then be restored with:

```rust
let (binarizer, generator) =
    RuleGenerator::load(
        "model.json",
        "model_data",
    )?;
```

This allows the generated rules and binarization state to be reused without repeating the training/search process.

---

# API

The main public types are re-exported by the crate:

```rust
use lad_rs::{Binarizer, RuleGenerator};
```

## `Binarizer`

```rust
Binarizer::new(
    threshold,
    nominal_size,
    max_cutpoints_per_column,
    depth,
)
```

| Parameter                  | Description                                                                      |
| -------------------------- | -------------------------------------------------------------------------------- |
| `threshold`                | Minimum score required for a numerical split                                     |
| `nominal_size`             | Maximum number of unique values for a numerical feature to be treated as nominal |
| `max_cutpoints_per_column` | Maximum number of cut points retained for a column                               |
| `depth`                    | Maximum recursive depth of numerical cut-point search                            |

Main methods:

```rust
generate_cutpoints()
transform()
get_cutpoints()
save()
load()
```

## `RuleGenerator`

```rust
RuleGenerator::new(
    &binarizer,
    max,
)
```

| Parameter   | Description                                                               |
| ----------- | ------------------------------------------------------------------------- |
| `binarizer` | Binarizer used for feature transformation                                 |
| `max`       | Maximum number of features in a generated pattern; `0` means all features |

Main methods:

```rust
fit()
fit_new()
fit_legacy()
predict()
predict_exact()
predict_fuzzy()
get_rules()
save()
load()
```

---

# Parallelism

Pattern coverage across classes is evaluated using [Rayon](https://github.com/rayon-rs/rayon).

Conceptually:

```text
                    Candidate Pattern
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Class 0       Class 1       Class 2
          coverage      coverage      coverage
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Coverage counts
```

This allows independent class-specific coverage calculations to run in parallel.

---

# Progress reporting

Long-running rule searches can emit desktop notifications using [`notify-rust`](https://crates.io/crates/notify-rust).

Progress information includes the current pattern-search stage and remaining observations.

This is particularly useful because LAD pattern generation can become computationally expensive as the feature count and search depth increase.

---

# Complexity considerations

LAD pattern generation is inherently combinatorial.

If there are `F` binary features and patterns can contain up to `d` terms, the number of possible combinations can grow rapidly with `F` and `d`.

Consequently, runtime is strongly affected by:

* Number of input samples
* Number of generated binary features
* Number of classes
* Maximum pattern size
* Numerical cut-point search depth
* Number of candidate patterns surviving each search stage

The implementation therefore exposes parameters such as `max`, `search_depth`, and `threshold` to control the search space.

Parallel class-wise coverage evaluation can reduce the cost of individual pattern evaluations, but it does not eliminate the combinatorial nature of the search.

---

# Example workflow

A typical workflow is:

```rust
use lad_rs::{Binarizer, RuleGenerator};

// Create the binarizer.
let mut binarizer = Binarizer::new(
    0.25,
    10,
    20,
    4,
);

// Learn the binary representation from training data.
binarizer.generate_cutpoints(
    &train_data,
    &train_labels,
)?;

// Create the rule generator.
let mut rules = RuleGenerator::new(
    &binarizer,
    4,
);

// Generate LAD rules.
let remaining = rules.fit(
    &train_data,
    &train_labels,
)?;

// Inspect generated rules.
for rule in rules.get_rules() {
    println!("{rule:?}");
}

// Classify new observations.
let predictions = rules.predict(
    &test_data,
)?;
```

---

# Project Structure

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

The library currently exposes:

```rust
pub mod binarization;

pub use crate::binarization::{
    binarize::Binarizer,
    rule_generation::RuleGenerator,
};
```

---

# Dependencies

| Crate                                               | Purpose                                |
| --------------------------------------------------- | -------------------------------------- |
| [Polars](https://pola.rs/)                          | DataFrame and columnar data processing |
| [Rayon](https://crates.io/crates/rayon)             | Parallel pattern evaluation            |
| [itertools](https://crates.io/crates/itertools)     | Iterator utilities                     |
| [Serde](https://serde.rs/)                          | Model serialization                    |
| [serde_json](https://crates.io/crates/serde_json)   | JSON model persistence                 |
| [notify-rust](https://crates.io/crates/notify-rust) | Desktop progress notifications         |

---

# Building

Clone the repository:

```bash
git clone https://github.com/Bha-Gu/lad_rs.git
cd lad_rs
```

Build:

```bash
cargo build
```

Build an optimized release:

```bash
cargo build --release
```

---

# Research Context

`lad_rs` was developed as a Rust implementation of the LAD methodology used in research on **LAD-based intrusion detection**.

The associated research work includes:

> **IDS-LAD: Intrusion detection system using logical analysis of data**

Bhaumikaditya Guleria, Maroti Deshmukh, and Rakhi Nautiyal, *Cluster Computing*, 2025.

The Rust implementation is intended to provide a native implementation of the underlying binarization and logical-pattern generation process.

---

# Status

This project is currently **experimental/research software**.

The implementation is actively oriented toward algorithmic experimentation, performance work, and application to classification problems.

The public API and internal search algorithms may change.

---

# License

See [`LICENSE`](LICENSE) for licensing information.
