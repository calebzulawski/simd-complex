#![feature(portable_simd)]

use num_complex as nc;
use simd_traits::{Num, Vector};

use core::{ops, simd::Simd};

#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Hash)]
#[repr(C)]
pub struct Complex<T> {
    re: T,
    im: T,
}

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

unsafe impl<T> Vector for Complex<T>
where
    T: Vector,
{
    type Scalar = nc::Complex<T::Scalar>;
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
        let re = self.re.clone() + rhs.re.clone();
        let im = self.im.clone() + rhs.im.clone();
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
        let im = self.im.clone() - rhs.im.clone();
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
        let im = self.re.clone() * rhs.im.clone() + self.im.clone() * rhs.re.clone();
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
        let im = (self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone()) / denom;
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
