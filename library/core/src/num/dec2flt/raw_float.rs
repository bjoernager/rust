//! Helper trait for generic float types.

use core::f128;

use crate::fmt::{Debug, LowerExp};
use crate::num::FpCategory;
use crate::ops::{self, Add, Div, Mul, Neg};

/// Lossy `as` casting between two types.
pub trait CastInto<T: Copy>: Copy {
    #[must_use]
    fn cast(self) -> T;
}

/// Collection of traits that allow us to be generic over integer size.
pub trait Integer:
    Sized
    + Clone
    + Copy
    + Debug
    + ops::Shr<u32, Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + PartialEq
    + CastInto<i16>
{
    /// The value zero.
    const ZERO: Self;

    /// The value one.
    const ONE: Self;
}

macro_rules! def_integers {
    ($($ty:ty)*) => {
        $(
            impl CastInto<i16> for $ty {
                #[inline]
                fn cast(self) -> i16 {
                    self as i16
                }
            }

            impl Integer for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
            }
        )+
    }
}

def_integers! { u16 u32 u64 u128 }

/// A helper trait to avoid duplicating basically all the conversion code for IEEE floats.
///
/// See the parent module's doc comment for why this is necessary.
///
/// Should **never, ever** be implemented for other types or be used outside the `dec2flt` module.
#[doc(hidden)]
pub unsafe trait RawFloat:
    Copy
    + Div<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Add<Output = Self>
    + LowerExp
    + PartialOrd
    + Default
    + Debug
{
    /// The unsigned integer with the same width as the float.
    type Bits: Integer + Into<u128>;

    /// Infinity.
    const INFINITY: Self;

    /// Negative infinity.
    const NEG_INFINITY: Self;

    /// Not-a-number.
    const NAN: Self;

    /// Not-a-number with a negative sign.
    const NEG_NAN: Self;

    /// Bit width of the float
    const BITS: u32;

    /// The number of bits in the significand, *including* the hidden bit.
    const MANTISSA_DIGITS: u32;

    /// An integer mask for the mantissa bits.
    const MANTISSA_MASK: Self::Bits;

    /// An integer mask for the exponent bits.
    const EXPONENT_MASK: Self::Bits;

    /// The number of bits in the significand, *excluding* the hidden bit.
    const MANTISSA_BITS: u32 = Self::MANTISSA_DIGITS - 1;

    /// Minimum exponent value of normal values.
    const MIN_EXPONENT: i32 = -(Self::EXPONENT_BIAS as i32 - 1);

    /// The saturated (maximum bitpattern) value of the exponent, i.e. the infinite
    /// representation.
    ///
    /// This shifted fully right, use `EXPONENT_MASK` for the shifted value.
    const EXPONENT_SATURATED: u32 = (1 << Self::EXPONENT_BITS) - 1;

    /// Signed version of `EXPONENT_SATURATED` since we convert a lot.
    const INFINITE_POWER: i32 = Self::EXPONENT_SATURATED as i32;

    /// Minimum exponent for a fast path case, or `-floor(MANTISSA_DIGITS / log2(5))`
    const MIN_EXPONENT_FAST_PATH: i128 = -Self::MAX_EXPONENT_FAST_PATH;

    /// Maximum exponent for a fast path case, or `floor(MANTISSA_DIGITS / log2(5))`,
    /// assuming `FLT_EVAL_METHOD == 0`.
    const MAX_EXPONENT_FAST_PATH: i128 = {
        let log2_5 = f128::consts::LOG2_10 - 1.0;
        (Self::MANTISSA_DIGITS as f128 / log2_5) as i128
    };

    /// Maximum exponent that can be represented for a disguised-fast path case.
    /// This is `MAX_EXPONENT_FAST_PATH + floor(MANTISSA_DIGITS / log2(10))`
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i128 =
        Self::MAX_EXPONENT_FAST_PATH + (Self::MANTISSA_DIGITS as f128 / f128::consts::LOG2_10) as i128;

    /// Maximum mantissa for the fast-path (`1 << 53` for `f128`).
    const MAX_MANTISSA_FAST_PATH: u128 = 1 << Self::MANTISSA_DIGITS;

    /// Converts integer into float through an `as` cast.
    /// This is only called in the fast-path algorithm, and therefore will not lose
    /// precision, since the value will always have only if the value is less than or
    /// equal to `Self::MAX_MANTISSA_FAST_PATH``.
    #[must_use]
    fn from_u128(v: u128) -> Self;

    /// Performs a raw transmutation from an integer.
    /// Extraneous bits are ignored.
    #[must_use]
    fn from_u128_bits(v: u128) -> Self;

    /// Gets a small power-of-ten for fast-path multiplication.
    #[must_use]
    fn pow10_fast_path(exponent: usize) -> Self;

    /// Returns the category that this number falls into.
    #[must_use]
    fn classify(self) -> FpCategory;

    /// Transmute to the integer representation
    #[must_use]
    fn to_bits(self) -> Self::Bits;

    /// Returns the mantissa, the exponent, and the sign, as integers.
    ///
    /// This returns `(m, p, s)` such that `s * m * 2^p` represents the original float.
    /// For 0, the exponent will be `-(EXPONENT_BIAS + MANTISSA_BITS)`, which is the
    /// minimum subnormal power.
    /// For infinity or NaN, the exponent will be
    /// `EXPONENT_SATURATED - EXPONENT_BIAS - MANTISSA_BITS`.
    ///
    /// If subnormal, the mantissa will be shifted one bit to the left.
    /// Otherwise, it is returned with the explicit bit set but otherwise unshifted.
    ///
    /// `s` is only ever &#177;1.
    #[must_use]
    fn integer_decode(self) -> (u128, i16, i8) {
        let bits = self.to_bits();

        let sign: i8 = if bits >> (Self::BITS - 1) == Self::Bits::ZERO { 1 } else { -1 };

        let mut exponent: i16 = ((bits & Self::EXPONENT_MASK) >> Self::MANTISSA_BITS).cast();

        let mantissa = if exponent == 0 {
            (bits & Self::MANTISSA_MASK) << 1
        } else {
            (bits & Self::MANTISSA_MASK) | (Self::Bits::ONE << Self::MANTISSA_BITS)
        };

        // Exponent bias + mantissa shift
        exponent -= (Self::EXPONENT_BIAS + Self::MANTISSA_BITS) as i16;

        (mantissa.into(), exponent, sign)
    }
}

#[cfg(target_has_reliable_f16)]
unsafe impl RawFloat for f16 {
    type Bits = u16;

    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self = Self::NEG_INFINITY;
    const NAN: Self = Self::NAN;
    const NEG_NAN: Self = -Self::NAN;

    const BITS: u32 = 16;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const EXPONENT_MASK: Self::Bits = Self::EXPONENT_MASK;
    const MANTISSA_MASK: Self::Bits = Self::MAN_MASK;

    #[inline]
    fn from_u128(v: u128) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u128_bits(v: u128) -> Self {
        Self::from_bits((v & 0xFFFF) as u16)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        const TABLE: [f16; 8] = [1e0, 1e1, 1e2, 1e3, 1e4, 0.0, 0.0, 0.0];

        TABLE[exponent & 7]
    }

    #[inline]
    fn to_bits(self) -> Self::Bits {
        self.to_bits()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.classify()
    }
}

unsafe impl RawFloat for f32 {
    type Bits = u32;

    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const NEG_NAN: Self = -f32::NAN;

    const BITS: u32 = 32;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const EXPONENT_MASK: Self::Bits = Self::EXP_MASK;
    const MANTISSA_MASK: Self::Bits = Self::MAN_MASK;

    #[inline]
    fn from_u128(v: u128) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u128_bits(v: u128) -> Self {
        f32::from_bits((v & 0xFFFFFFFF) as u32)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        const TABLE: [f32; 16] = [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        TABLE[exponent & 15]
    }

    #[inline]
    fn to_bits(self) -> Self::Bits {
        self.to_bits()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.classify()
    }
}

unsafe impl RawFloat for f64 {
    type Bits = u64;

    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self = Self::NEG_INFINITY;
    const NAN: Self = Self::NAN;
    const NEG_NAN: Self = -Self::NAN;

    const BITS: u32 = 64;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const EXPONENT_MASK: Self::Bits = Self::EXPONENT_MASK;
    const MANTISSA_MASK: Self::Bits = Self::MAN_MASK;

    #[inline]
    fn from_u128(v: u128) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u128_bits(v: u128) -> Self {
        Self::from_bits((v & 0xFFFFFFFF) as u64)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        const TABLE: [f64; 32] = [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
            1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        TABLE[exponent & 31]
    }

    #[inline]
    fn to_bits(self) -> Self::Bits {
        self.to_bits()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.classify()
    }
}

#[cfg(target_has_reliable_f128)]
unsafe impl RawFloat for f128 {
    type Bits = u128;

    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self = Self::NEG_INFINITY;
    const NAN: Self = Self::NAN;
    const NEG_NAN: Self = -Self::NAN;

    const BITS: u32 = 128;
    const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
    const EXPONENT_MASK: Self::Bits = Self::EXPONENT_MASK;
    const MANTISSA_MASK: Self::Bits = Self::MAN_MASK;

    #[inline]
    fn from_u128(v: u128) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u128_bits(v: u128) -> Self {
        Self::from_bits(v)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        const TABLE: [f128; 64] = [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
            1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31,
            1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39, 1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47,
            1e48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        TABLE[exponent & 63]
    }

    #[inline]
    fn to_bits(self) -> Self::Bits {
        self.to_bits()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.classify()
    }
}
