//! Safetensors file format parser.
//!
//! The safetensors format is designed for zero-copy loading:
//! - 8-byte little-endian header size
//! - JSON header with tensor metadata (name, dtype, shape, data offsets)
//! - Raw tensor data (directly memory-mappable)
//!
//! This parser reads the JSON header and provides byte offsets into the
//! memory-mapped file for each tensor, enabling zero-copy MLX array creation.

use std::collections::HashMap;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use serde::Deserialize;

use crate::error::{LlamaError, Result};
use crate::mlx::dtype::Dtype;
use crate::weights::mmap::MappedFile;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    /// Data type string (e.g., "F16", "F32", "BF16", "I32", "U32").
    pub dtype: String,

    /// Tensor shape.
    pub shape: Vec<usize>,

    /// Byte offset range `[start, end)` within the data section.
    pub data_offsets: [usize; 2],
}

impl TensorInfo {
    /// Parse the dtype string into our Dtype enum.
    pub fn parse_dtype(&self) -> Result<Dtype> {
        match self.dtype.as_str() {
            "F32" => Ok(Dtype::Float32),
            "F16" => Ok(Dtype::Float16),
            "BF16" => Ok(Dtype::BFloat16),
            "I32" => Ok(Dtype::Int32),
            "I64" => Ok(Dtype::Int64),
            "I16" => Ok(Dtype::Int16),
            "I8" => Ok(Dtype::Int8),
            "U8" => Ok(Dtype::UInt8),
            "U16" => Ok(Dtype::UInt16),
            "U32" => Ok(Dtype::UInt32),
            "BOOL" => Ok(Dtype::Bool),
            other => Err(LlamaError::UnsupportedDtype(other.to_string())),
        }
    }

    /// Total byte size of the tensor data.
    pub fn byte_size(&self) -> usize {
        self.data_offsets[1] - self.data_offsets[0]
    }

    /// Shape as i32 for MLX.
    pub fn shape_i32(&self) -> Vec<i32> {
        self.shape.iter().map(|&s| s as i32).collect()
    }
}

/// Parsed safetensors file header.
pub struct SafetensorsHeader {
    /// Map from tensor name to metadata.
    pub tensors: HashMap<String, TensorInfo>,

    /// Optional metadata (e.g., quantization info).
    pub metadata: HashMap<String, String>,

    /// Byte offset where the data section begins (after header).
    pub data_offset: usize,
}

/// Parse a safetensors file header from a memory-mapped file.
///
/// # Format
/// ```text
/// [8 bytes: header_size (u64 LE)]
/// [header_size bytes: JSON header]
/// [remaining bytes: tensor data]
/// ```
pub fn parse_header(mapped: &MappedFile) -> Result<SafetensorsHeader> {
    let bytes = mapped.as_bytes();

    if bytes.len() < 8 {
        return Err(LlamaError::WeightLoad("file too small for safetensors header".into()));
    }

    // Read 8-byte little-endian header size
    let header_size = (&bytes[..8]).read_u64::<LittleEndian>().map_err(|e| {
        LlamaError::WeightLoad(format!("failed to read header size: {e}"))
    })? as usize;

    if 8 + header_size > bytes.len() {
        return Err(LlamaError::WeightLoad(format!(
            "header size {header_size} exceeds file size {}", bytes.len()
        )));
    }

    let header_json = &bytes[8..8 + header_size];
    let header_str = std::str::from_utf8(header_json).map_err(|e| {
        LlamaError::WeightLoad(format!("invalid UTF-8 in header: {e}"))
    })?;

    // Parse JSON: the header is a map of tensor_name -> TensorInfo,
    // with an optional "__metadata__" key.
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)?;

    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();

    for (key, value) in raw {
        if key == "__metadata__" {
            // Metadata is a flat string->string map
            if let Some(obj) = value.as_object() {
                for (mk, mv) in obj {
                    if let Some(s) = mv.as_str() {
                        metadata.insert(mk.clone(), s.to_string());
                    }
                }
            }
        } else {
            let info: TensorInfo = serde_json::from_value(value).map_err(|e| {
                LlamaError::WeightLoad(format!("failed to parse tensor '{key}': {e}"))
            })?;
            tensors.insert(key, info);
        }
    }

    Ok(SafetensorsHeader {
        tensors,
        metadata,
        data_offset: 8 + header_size,
    })
}

/// A loaded safetensors file with parsed header and memory-mapped data.
pub struct SafetensorsFile {
    /// Parsed header with tensor metadata.
    pub header: SafetensorsHeader,

    /// Memory-mapped file data.
    pub mapped: MappedFile,
}

impl SafetensorsFile {
    /// Open and parse a safetensors file.
    pub fn open(path: &Path) -> Result<Self> {
        let mapped = MappedFile::open(path)?;
        let header = parse_header(&mapped)?;
        Ok(SafetensorsFile { header, mapped })
    }

    /// Get the raw bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.header.tensors.get(name).ok_or_else(|| {
            LlamaError::WeightLoad(format!("tensor '{name}' not found in safetensors"))
        })?;
        let start = self.header.data_offset + info.data_offsets[0];
        let len = info.byte_size();
        self.mapped.slice(start, len)
    }

    /// Get a raw pointer to tensor data for zero-copy MLX array creation.
    pub fn tensor_ptr(&self, name: &str) -> Result<(*const u8, &TensorInfo)> {
        let info = self.header.tensors.get(name).ok_or_else(|| {
            LlamaError::WeightLoad(format!("tensor '{name}' not found in safetensors"))
        })?;
        let offset = self.header.data_offset + info.data_offsets[0];
        Ok((self.mapped.ptr_at(offset), info))
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }
}
