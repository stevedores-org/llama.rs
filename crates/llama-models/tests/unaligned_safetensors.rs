use llama_models::ModelWeights;

#[test]
fn load_unaligned_safetensors() {
    // Construct a safetensors file with unaligned data offset.
    // 8 bytes (u64) + header_len + data.
    // We want (8 + header_len) % 4 != 0.

    // Minimal valid header content
    let header_json = r#"{"test":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut header_bytes = header_json.as_bytes().to_vec();

    // Current length
    let len = header_bytes.len();
    // We want (8 + len + padding) % 4 == 1 (or 2, or 3).
    // Let's target 1.
    // (8 + len + p) % 4 = 1
    // (len + p) % 4 = 1

    let remainder = len % 4;
    let padding_needed = if remainder <= 1 {
        1 - remainder
    } else {
        1 + 4 - remainder
    };

    // Add spaces to padding (valid JSON whitespace)
    header_bytes.extend(std::iter::repeat_n(b' ', padding_needed));

    let header_len = header_bytes.len();
    assert_eq!(
        (8 + header_len) % 4,
        1,
        "Header should end at unaligned offset"
    );

    let mut file_bytes = Vec::new();
    // u64 header length (little endian)
    file_bytes.extend_from_slice(&(header_len as u64).to_le_bytes());
    // header
    file_bytes.extend_from_slice(&header_bytes);
    // data (4 bytes for one f32)
    let data: f32 = 123.456;
    file_bytes.extend_from_slice(&data.to_le_bytes()); // 4 bytes

    // Verify alignment of data in the buffer
    let data_offset = 8 + header_len;
    assert_eq!(data_offset % 4, 1);

    // The buffer itself 'file_bytes' is aligned in memory (Vec<u8> is aligned to 1, but its pointer is aligned).
    // But we are passing &file_bytes which points to aligned memory.
    // The data inside is at offset data_offset.

    // Attempt to load
    let result = ModelWeights::load_safetensors_bytes(&file_bytes);

    // This should succeed if fixed, or panic/error if not.
    // We expect it to SUCCEED with the fix.
    // Before fix, it panics.
    assert!(
        result.is_ok(),
        "Loading unaligned safetensors failed: {:?}",
        result.err()
    );

    let weights = result.unwrap();
    let tensor = weights.get("test").unwrap();
    assert_eq!(tensor.data[0], 123.456);
}
