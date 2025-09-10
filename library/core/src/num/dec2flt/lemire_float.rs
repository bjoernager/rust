//! Helper trait for generic float types.

use core::f64;

use crate::num::dec2flt::raw_float::{RawFloat};

/// Denotes a [`RawFloat`] type that is usable in the Eisel-Lemire algorithm.
#[doc(hidden)]
pub unsafe trait LemireFloat: RawFloat {
    /// The exponent bias value.
    ///
    /// This is also the maximum value of the exponent.
    const EXPONENT_BIAS: u32 = Self::EXPONENT_SATURATED >> 1;

    // Round-to-even only happens for negative values of q
    // when q >= −4 in the 64-bit case and when q >= −17 in
    // the 32-bit case.
    //
    // When `q >= 0`, we have 5^q <= 2m+1. In the 64-bit case,we
    // have 5^q <= 2m+1 <= 2^54 or q <= 23. In the 32-bit case,we have
    // 5^q <= 2m+1 <= 2^25 or q <= 10.
    //
    // When q < 0, we have w >= (2m+1)×5^−q. We must have that w < 2^64
    // so (2m+1)×5^−q < 2^64. We have that 2m+1 > 2^53 (64-bit case)
    // or 2m+1 > 2^24 (32-bit case). Hence,we must have 2^53×5^−q < 2^64
    // (64-bit) and 2^24×5^−q < 2^64 (32-bit). Hence we have 5^−q < 2^11
    // or q >= −4 (64-bit case) and 5^−q < 2^40 or q >= −17 (32-bit case).
    //
    // Thus we have that we only need to round ties to even when
    // we have that q ∈ [−4,23](in the 64-bit case) or q∈[−17,10]
    // (in the 32-bit case). In both cases,the power of five(5^|q|)
    // fits in a 64-bit word.

    const MIN_EXPONENT_ROUND_TO_EVEN: i32;

    const MAX_EXPONENT_ROUND_TO_EVEN: i32;

    /// Smallest decimal exponent for a non-zero value.
    /// This allows for fast pathing anything smaller than `10 ^ SMALLEST_POWER_OF_TEN`,
    /// which will round to zero.
    ///
    /// The smallest power of ten is represented by
    /// `floor(log10(2 ^ (-n) / (2 ^ 128 - 1)))`, where `n` is the smallest power of
    /// two.
    /// The `(2^128 - 1)` denominator comes from the maximum value that is representable
    /// by the intermediate storage format. We don't actually know *why* the storage
    /// format is relevant here.
    ///
    /// The values may be calculated using the formula. Unfortunately we cannot
    /// calculate them at compile time since intermediates exceed the range of an `f64`.
    const SMALLEST_POWER_OF_TEN: i32;

    /// Largest, decimal exponent for a finite value.
    ///
    /// This is the max exponent in binary converted to the max exponent in decimal.
    /// Allows fast pathing anything larger than `10 ^ LARGEST_POWER_OF_TEN`, which will
    /// round to infinity.
    const LARGEST_POWER_OF_TEN: i32 = {
        let largest_power_of_two = Self::EXPONENT_BIAS + 1;
        pow2_to_pow10(largest_power_of_two as i64) as i32
    };

    /// Maximum exponent that can be represented for a disguised-fast path case.
    /// This is `MAX_EXPONENT_FAST_PATH + floor(MANTISSA_DIGITS / log2(10))`
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i64 =
        Self::MAX_EXPONENT_FAST_PATH as i64 + (Self::MANTISSA_DIGITS as f64 / f64::consts::LOG2_10) as i64;

    /// Maximum mantissa for the fast-path (`1 << 53` for `f64`).
    const MAX_MANTISSA_FAST_PATH: u64 = 1 << Self::MANTISSA_DIGITS;

    /// Converts integer into float through an `as` cast.
    /// This is only called in the fast-path algorithm, and therefore will not lose
    /// precision, since the value will always have only if the value is less than or
    /// equal to `Self::MAX_MANTISSA_FAST_PATH``.
    #[must_use]
    fn from_u64(v: u64) -> Self;

    /// Performs a raw transmutation from an integer.
    /// Extraneous bits are ignored.
    #[must_use]
    fn from_u64_bits(v: u64) -> Self;
}

/// Solves for `b` in `10 ^ b == 2 ^ a`.
#[inline]
#[must_use]
const fn pow2_to_pow10(a: i64) -> i64 {
    let res = (a as f64) / f64::consts::LOG2_10;
    res as i64
}

#[cfg(target_has_reliable_f16)]
unsafe impl LemireFloat for f16 {
    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -22;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 5;

    const SMALLEST_POWER_OF_TEN: i32 = -27;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        Self::from_bits((v & 0xFFFF) as u16)
    }
}

unsafe impl LemireFloat for f32 {
    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -17;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 10;

    const SMALLEST_POWER_OF_TEN: i32 = -65;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        f32::from_bits((v & 0xFFFFFFFF) as u32)
    }
}

unsafe impl LemireFloat for f64 {
    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -4;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 23;

    const SMALLEST_POWER_OF_TEN: i32 = -342;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        Self::from_bits((v & 0xFFFFFFFF) as u64)
    }
}
