# Task: i8_roll_add — Circular Roll then Add

## Function signature

```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k);
```

## Semantics

Circularly shift array `a` by `k` positions, then add element-wise with `b`:

```
out[i] = (int8_t)( a[(i + k) % n] + b[i] )
```

The addition is **two's-complement wraparound** (not saturating).
`k` is a runtime parameter — do NOT hardcode it.

## Key rules

- The circular index must wrap modulo `n` (not modulo 128 or any fixed stride).
- `k` can be any value in `[0, n-1]`; the harness sweeps several values.
- Addition wraps: `127 + 1 == -128` (int8 overflow is intentional here).
- `n` is a runtime parameter.

## HVX hints

The non-contiguous gather (`(i+k) % n`) breaks simple vector loads; the practical
approach is to split into two contiguous segments:

```
seg1: a[k .. n-1]  -> out[0 .. n-1-k]
seg2: a[0 .. k-1]  -> out[n-k .. n-1]
```

then vector-add `b` to each segment.

## Example (n=8, k=3)

```
a = [10, 20, 30, 40, 50, 60, 70, 80]
b = [ 1,  2,  3,  4,  5,  6,  7,  8]
rolled_a = [40, 50, 60, 70, 80, 10, 20, 30]
out      = [41, 52, 63, 74, 85, 16, 27, 38]
```
