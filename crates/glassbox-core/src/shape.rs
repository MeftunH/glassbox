use smallvec::{smallvec, SmallVec};

use crate::error::{CoreError, Result};

const MAX_RANK: usize = 8;

#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape(SmallVec<[usize; MAX_RANK]>);

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().collect())
    }

    pub fn scalar() -> Self {
        Self(SmallVec::new())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn numel(&self) -> usize {
        self.0.iter().copied().product::<usize>().max(if self.0.is_empty() { 1 } else { 0 })
    }

    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.0
            .get(axis)
            .copied()
            .ok_or(CoreError::AxisOutOfBounds { axis, rank: self.rank() })
    }

    pub fn contiguous_strides(&self) -> Stride {
        let mut strides: SmallVec<[usize; MAX_RANK]> = smallvec![0; self.rank()];
        if self.rank() == 0 {
            return Stride(strides);
        }
        let mut acc = 1usize;
        for i in (0..self.rank()).rev() {
            strides[i] = acc;
            acc *= self.0[i];
        }
        Stride(strides)
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Self(SmallVec::from_slice(&v))
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self(SmallVec::from_slice(v))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(v.into())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stride(SmallVec<[usize; MAX_RANK]>);

impl Stride {
    pub fn new(strides: impl IntoIterator<Item = usize>) -> Self {
        Self(strides.into_iter().collect())
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    pub fn is_contiguous_for(&self, shape: &Shape) -> bool {
        &shape.contiguous_strides() == self
    }
}

impl std::fmt::Debug for Stride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        let st = s.contiguous_strides();
        assert_eq!(st.as_slice(), &[12, 4, 1]);
        assert!(st.is_contiguous_for(&s));
    }

    #[test]
    fn scalar_has_numel_one() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn axis_bounds() {
        let s = Shape::from([5, 6]);
        assert_eq!(s.dim(0).unwrap(), 5);
        assert_eq!(s.dim(1).unwrap(), 6);
        assert!(matches!(
            s.dim(2),
            Err(CoreError::AxisOutOfBounds { axis: 2, rank: 2 })
        ));
    }
}
