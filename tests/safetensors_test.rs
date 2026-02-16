//! Tests for safetensors parsing.

use std::io::Write;

use llama::weights::safetensors::{SafetensorsFile, TensorInfo};

#[test]
fn test_tensor_info_parse_dtype() {
    let info = TensorInfo {
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        data_offsets: [0, 800],
    };
    let dtype = info.parse_dtype().unwrap();
    assert_eq!(dtype, llama::mlx::dtype::Dtype::Float32);
    assert_eq!(info.byte_size(), 800);
    assert_eq!(info.shape_i32(), vec![10, 20]);
}

#[test]
fn test_tensor_info_dtypes() {
    let dtypes = vec![
        ("F16", llama::mlx::dtype::Dtype::Float16),
        ("F32", llama::mlx::dtype::Dtype::Float32),
        ("BF16", llama::mlx::dtype::Dtype::BFloat16),
        ("I32", llama::mlx::dtype::Dtype::Int32),
        ("U8", llama::mlx::dtype::Dtype::UInt8),
    ];

    for (name, expected) in dtypes {
        let info = TensorInfo {
            dtype: name.to_string(),
            shape: vec![1],
            data_offsets: [0, 4],
        };
        assert_eq!(info.parse_dtype().unwrap(), expected);
    }
}

#[test]
fn test_tensor_info_unsupported_dtype() {
    let info = TensorInfo {
        dtype: "INVALID".to_string(),
        shape: vec![1],
        data_offsets: [0, 4],
    };
    assert!(info.parse_dtype().is_err());
}

#[test]
fn test_parse_safetensors_file() {
    // Create a minimal safetensors file
    let header = serde_json::json!({
        "test_tensor": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        },
        "__metadata__": {
            "format": "pt"
        }
    });

    let header_str = serde_json::to_string(&header).unwrap();
    let header_bytes = header_str.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Build the file: 8-byte header size + header JSON + data
    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    // Add dummy tensor data (6 f32 values = 24 bytes)
    file_data.extend_from_slice(&[0u8; 24]);

    // Write to temp file and parse
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("test.safetensors");
    let mut f = std::fs::File::create(&file_path).unwrap();
    f.write_all(&file_data).unwrap();
    drop(f);

    let sf = SafetensorsFile::open(&file_path).unwrap();

    assert!(sf.header.tensors.contains_key("test_tensor"));
    assert_eq!(sf.header.tensors.len(), 1);
    assert_eq!(sf.header.metadata.get("format").unwrap(), "pt");

    let info = &sf.header.tensors["test_tensor"];
    assert_eq!(info.dtype, "F32");
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.byte_size(), 24);

    // Test data access
    let data = sf.tensor_data("test_tensor").unwrap();
    assert_eq!(data.len(), 24);

    // Test missing tensor
    assert!(sf.tensor_data("nonexistent").is_err());

    // Test tensor listing
    let names = sf.tensor_names();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"test_tensor"));
}
