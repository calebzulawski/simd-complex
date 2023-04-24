//! SIMD vectors of complex numbers.

#![feature(portable_simd)]

use core::{
    ops,
    simd::{Simd, SimdPartialEq},
};
use num_complex as nc;
use simd_traits::{
    num::Num,
    swizzle::{Shuffle, Swizzle, SwizzleIndex},
    Mask, Vector,
};

/// A vector of complex numbers.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Hash)]
#[repr(C)]
pub struct Complex<T> {
    re: T,
    im: T,
}

/// A vector of complex numbers, backed by [`std::simd::Simd`].
pub type SimdComplex<T, const N: usize> = Complex<Simd<T, N>>;

impl<T> Complex<T>
where
    T: Clone + Num,
{
    /// Return the imaginary unit `i`.
    pub fn i() -> Self {
        Self {
            re: T::zero(),
            im: T::one(),
        }
    }

    /// Return the square of the complex norm.
    pub fn norm_sqr(&self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl<T> SimdPartialEq for Complex<T>
where
    T: SimdPartialEq,
    T::Mask: Mask,
{
    type Mask = T::Mask;

    fn simd_eq(self, other: Self) -> Self::Mask {
        self.re.simd_eq(other.re) & self.im.simd_eq(other.im)
    }

    fn simd_ne(self, other: Self) -> Self::Mask {
        !self.simd_eq(other)
    }
}

unsafe impl<T> Vector for Complex<T>
where
    T: Vector,
{
    type Scalar = nc::Complex<T::Scalar>;
    type Mask = T::Mask;
    const ELEMENTS: usize = T::ELEMENTS;

    fn splat(value: Self::Scalar) -> Self {
        Self {
            re: T::splat(value.re),
            im: T::splat(value.im),
        }
    }

    unsafe fn extract_unchecked(&self, index: usize) -> Self::Scalar {
        let re = self.re.extract_unchecked(index);
        let im = self.im.extract_unchecked(index);
        Self::Scalar { re, im }
    }

    unsafe fn insert_unchecked(&mut self, index: usize, value: Self::Scalar) {
        self.re.insert_unchecked(index, value.re);
        self.im.insert_unchecked(index, value.im);
    }

    fn select(mask: &Self::Mask, true_values: Self, false_values: Self) -> Self {
        Complex {
            re: mask.select(true_values.re, false_values.re),
            im: mask.select(true_values.im, false_values.im),
        }
    }
}

impl<T> Shuffle for Complex<T>
where
    T: Shuffle,
{
    fn reverse(self) -> Self {
        Self {
            re: self.re.reverse(),
            im: self.im.reverse(),
        }
    }

    fn interleave(self, other: Self) -> (Self, Self) {
        let (re1, re2) = self.re.interleave(other.re);
        let (im1, im2) = self.im.interleave(other.im);
        (Self { re: re1, im: im1 }, Self { re: re2, im: im2 })
    }

    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (re1, re2) = self.re.deinterleave(other.re);
        let (im1, im2) = self.im.deinterleave(other.im);
        (Self { re: re1, im: im1 }, Self { re: re2, im: im2 })
    }
}

impl<T, U> Swizzle<Complex<U>> for Complex<T>
where
    T: Swizzle<U>,
    U: Vector<Scalar = T::Scalar>,
{
    fn swizzle<I: SwizzleIndex>(self) -> Complex<U> {
        Complex {
            re: self.re.swizzle::<I>(),
            im: self.im.swizzle::<I>(),
        }
    }
}

impl<T> Num for Complex<T>
where
    T: Clone + Num,
{
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    fn one() -> Self {
        Self {
            re: T::one(),
            im: T::zero(),
        }
    }
}

impl<T> From<T> for Complex<T>
where
    T: Num,
{
    fn from(re: T) -> Self {
        Self { re, im: T::zero() }
    }
}

impl<T> ops::Add for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let re = self.re + rhs.re;
        let im = self.im + rhs.im;
        Self::Output { re, im }
    }
}

impl<T> ops::Sub for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let re = self.re.clone() - rhs.re.clone();
        let im = self.im - rhs.im;
        Self { re, im }
    }
}

impl<T> ops::Mul for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re * rhs.im + self.im * rhs.re;
        Self { re, im }
    }
}

impl<T> ops::Div for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.norm_sqr();
        let re =
            (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone()) / denom.clone();
        let im = (self.im * rhs.re - self.re * rhs.im) / denom;
        Self { re, im }
    }
}

impl<T> ops::Rem for Complex<T>
where
    T: Clone + Num,
{
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        let div = self.clone() / rhs.clone();
        let trunc = Self {
            re: div.re.clone() - div.re % T::one(),
            im: div.im.clone() - div.im % T::one(),
        };
        self - rhs * trunc
    }
}

macro_rules! derived_impls {
    { $trait:ident :: $op:ident, $trait_assign:ident :: $op_assign:ident } => {
        impl<T> ops::$trait<Complex<T>> for &Complex<T>
        where
            T: Clone + Num,
        {
            type Output = Complex<T>;
            fn $op(self, rhs: Complex<T>) -> Self::Output {
                ops::$trait::$op(self.clone(), rhs)
            }
        }

        impl<T> ops::$trait<&Complex<T>> for Complex<T>
        where
            T: Clone + Num,
        {
            type Output = Complex<T>;
            fn $op(self, rhs: &Complex<T>) -> Self::Output {
                ops::$trait::$op(self, rhs.clone())
            }
        }

        impl<T> ops::$trait for &Complex<T>
        where
            T: Clone + Num,
        {
            type Output = Complex<T>;
            fn $op(self, rhs: Self) -> Self::Output {
                ops::$trait::$op(self.clone(), rhs.clone())
            }
        }

        impl<T> ops::$trait_assign for Complex<T>
        where
            T: Clone + Num,
        {
            fn $op_assign(&mut self, rhs: Self) {
                *self = ops::$trait::$op(self.clone(), rhs);
            }
        }

        impl<T> ops::$trait_assign<&Complex<T>> for Complex<T>
        where
            T: Clone + Num,
        {
            fn $op_assign(&mut self, rhs: &Complex<T>) {
                ops::$trait_assign::$op_assign(self, rhs.clone())
            }
        }
    }
}

derived_impls! { Add::add, AddAssign::add_assign }
derived_impls! { Sub::sub, SubAssign::sub_assign }
derived_impls! { Mul::mul, MulAssign::mul_assign }
derived_impls! { Div::div, DivAssign::div_assign }
derived_impls! { Rem::rem, RemAssign::rem_assign }
