//! Memory-mapped file I/O for zero-copy weight loading.
//!
//! Uses `mmap` to map safetensors files directly into the process address space.
//! On Apple Silicon with UMA, this allows the GPU to read weight data directly
//! from the mapped pages without any explicit data transfer.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{LlamaError, Result};

/// A memory-mapped file handle.
///
/// The mapped region remains valid for the lifetime of this struct.
/// Dropping it unmaps the file.
pub struct MappedFile {
    /// The memory map handle.
    mmap: Mmap,

    /// Size of the file in bytes.
    size: usize,
}

impl MappedFile {
    /// Map a file into memory.
    ///
    /// # Safety
    /// The file must not be modified while mapped. This is safe for
    /// safetensors files which are read-only model weights.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            LlamaError::WeightLoad(format!("failed to open {}: {e}", path.display()))
        })?;

        let metadata = file.metadata().map_err(|e| {
            LlamaError::WeightLoad(format!("failed to read metadata for {}: {e}", path.display()))
        })?;

        let size = metadata.len() as usize;

        // Safety: we treat the file as read-only and it won't be modified externally.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                LlamaError::WeightLoad(format!("failed to mmap {}: {e}", path.display()))
            })?
        };

        Ok(MappedFile { mmap, size })
    }

    /// Get the full mapped data as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get a subslice at the given offset and length.
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.size {
            return Err(LlamaError::WeightLoad(format!(
                "slice [{offset}..{}] exceeds file size {}", offset + len, self.size,
            )));
        }
        Ok(&self.mmap[offset..offset + len])
    }

    /// Get a raw pointer to data at the given offset.
    ///
    /// Used for creating zero-copy MLX arrays from mapped weight data.
    pub fn ptr_at(&self, offset: usize) -> *const u8 {
        unsafe { self.mmap.as_ptr().add(offset) }
    }

    /// Total file size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}
