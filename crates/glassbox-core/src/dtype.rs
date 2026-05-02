#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    U8,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::U8 => "u8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_consistent() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::BF16.size(), 2);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::U32.size(), 4);
        assert_eq!(DType::U8.size(), 1);
    }

    #[test]
    fn float_classification() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::U32.is_float());
        assert!(!DType::U8.is_float());
    }
}
