//! Tests for MLX dtype definitions.

use llama::mlx::dtype::Dtype;

#[test]
fn test_dtype_size() {
    assert_eq!(Dtype::Bool.size_bytes(), 1);
    assert_eq!(Dtype::UInt8.size_bytes(), 1);
    assert_eq!(Dtype::Float16.size_bytes(), 2);
    assert_eq!(Dtype::BFloat16.size_bytes(), 2);
    assert_eq!(Dtype::Float32.size_bytes(), 4);
    assert_eq!(Dtype::Int32.size_bytes(), 4);
    assert_eq!(Dtype::Int64.size_bytes(), 8);
}

#[test]
fn test_dtype_roundtrip() {
    for code in 0..=11 {
        let dtype = Dtype::from_raw(code).unwrap();
        assert_eq!(dtype.to_raw(), code);
    }
}

#[test]
fn test_dtype_invalid() {
    assert!(Dtype::from_raw(99).is_err());
    assert!(Dtype::from_raw(-1).is_err());
}

#[test]
fn test_dtype_classification() {
    assert!(Dtype::Float32.is_float());
    assert!(Dtype::Float16.is_float());
    assert!(Dtype::BFloat16.is_float());
    assert!(!Dtype::Int32.is_float());

    assert!(Dtype::Int32.is_integer());
    assert!(Dtype::UInt8.is_integer());
    assert!(!Dtype::Float32.is_integer());
}

#[test]
fn test_dtype_names() {
    assert_eq!(Dtype::Float32.name(), "float32");
    assert_eq!(Dtype::Float16.name(), "float16");
    assert_eq!(Dtype::BFloat16.name(), "bfloat16");
    assert_eq!(Dtype::Int32.name(), "int32");
}
