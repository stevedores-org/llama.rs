//! MLX data type definitions.
//!
//! Maps MLX's internal dtype codes to a Rust enum for type-safe operations.

use crate::error::{LlamaError, Result};

/// Supported MLX data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Dtype {
    Bool = 0,
    UInt8 = 1,
    UInt16 = 2,
    UInt32 = 3,
    Int8 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    Float16 = 8,
    Float32 = 9,
    BFloat16 = 10,
    Complex64 = 11,
}

impl Dtype {
    /// Size of a single element in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::Bool | Dtype::UInt8 | Dtype::Int8 => 1,
            Dtype::UInt16 | Dtype::Int16 | Dtype::Float16 | Dtype::BFloat16 => 2,
            Dtype::UInt32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Int64 | Dtype::Complex64 => 8,
        }
    }

    /// Convert from raw integer code returned by MLX C API.
    pub fn from_raw(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Dtype::Bool),
            1 => Ok(Dtype::UInt8),
            2 => Ok(Dtype::UInt16),
            3 => Ok(Dtype::UInt32),
            4 => Ok(Dtype::Int8),
            5 => Ok(Dtype::Int16),
            6 => Ok(Dtype::Int32),
            7 => Ok(Dtype::Int64),
            8 => Ok(Dtype::Float16),
            9 => Ok(Dtype::Float32),
            10 => Ok(Dtype::BFloat16),
            11 => Ok(Dtype::Complex64),
            _ => Err(LlamaError::UnsupportedDtype(format!("unknown dtype code: {code}"))),
        }
    }

    /// Convert to the raw integer code expected by the MLX C API.
    pub fn to_raw(self) -> i32 {
        self as i32
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "bool",
            Dtype::UInt8 => "uint8",
            Dtype::UInt16 => "uint16",
            Dtype::UInt32 => "uint32",
            Dtype::Int8 => "int8",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::BFloat16 => "bfloat16",
            Dtype::Complex64 => "complex64",
        }
    }

    /// Whether this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::BFloat16 | Dtype::Complex64
        )
    }

    /// Whether this is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Dtype::UInt8
                | Dtype::UInt16
                | Dtype::UInt32
                | Dtype::Int8
                | Dtype::Int16
                | Dtype::Int32
                | Dtype::Int64
        )
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
