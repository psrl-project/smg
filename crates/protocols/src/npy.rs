//! Minimal NumPy `.npy` v1.0 writer.
//!
//! Used at the HTTP/JSON boundary to serialize `routed_experts` payloads in a
//! format identical to vLLM's HTTP responses, so existing trainers can reuse
//! the same `np.load(io.BytesIO(...))` decoder.
//!
//! # Format
//!
//! ```text
//!   "\x93NUMPY"  6 bytes magic
//!   \x01\x00     2 bytes version (1.0)
//!   <header_len> 2 bytes little-endian u16  (length of dict text + padding)
//!   <header>     ASCII dict, e.g. "{'descr': '|u1', 'fortran_order': False, 'shape': (12, 64, 8), }"
//!                Padded with spaces and terminated with '\n' so the start of the
//!                raw data block is aligned to 64 bytes from start of file.
//!   <data>       Raw C-contiguous bytes
//! ```
//!
//! Only the dtypes we currently emit are supported (`uint8`, `uint16`); add
//! more variants when models with larger expert spaces land.

const MAGIC: &[u8] = b"\x93NUMPY";
const VERSION: [u8; 2] = [0x01, 0x00];
/// .npy v1.0 aligns the start of the data block to a 64-byte boundary.
const ALIGNMENT: usize = 64;

/// NumPy dtype descriptor for the supported width set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    /// `uint8` — used when the model has ≤ 256 experts.
    U8,
    /// `uint16` — used when the model has > 256 experts.
    U16,
    /// `float32` — reserved for future per-token scalar exports.
    F32,
}

impl NpyDtype {
    /// The dtype string written into the .npy header.
    fn descr(self) -> &'static str {
        match self {
            // Endianness is irrelevant for single-byte dtypes; use `|u1`.
            Self::U8 => "|u1",
            Self::U16 => "<u2",
            Self::F32 => "<f4",
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Encode a C-contiguous byte buffer as a NumPy `.npy` v1.0 file.
///
/// `shape` is row-major; `data.len()` must equal `shape.product() * dtype.size()`.
pub fn encode_npy(shape: &[u64], dtype: NpyDtype, data: &[u8]) -> Vec<u8> {
    let header = build_header(shape, dtype);
    let mut out = Vec::with_capacity(header.len() + data.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    out
}

fn build_header(shape: &[u64], dtype: NpyDtype) -> Vec<u8> {
    let shape_str = format_shape(shape);
    // Python dict literal as defined by the .npy spec; trailing comma matches
    // numpy.lib.format.write_array_header_1_0.
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype.descr(),
        shape_str,
    );

    // Compute padding so that the total prefix length (magic + version +
    // header_len field + dict + padding + '\n') is a multiple of 64.
    let prefix_fixed = MAGIC.len() + VERSION.len() + 2; // 10 bytes
    let unpadded = prefix_fixed + dict.len() + 1; // +1 for '\n'
    let padded = unpadded.div_ceil(ALIGNMENT) * ALIGNMENT;
    let pad_spaces = padded - unpadded;

    let header_text_len = dict.len() + pad_spaces + 1; // dict + padding + '\n'
    debug_assert!(
        header_text_len <= u16::MAX as usize,
        "header too large for npy v1.0"
    );

    let mut out = Vec::with_capacity(prefix_fixed + header_text_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION);
    out.extend_from_slice(&(header_text_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad_spaces));
    out.push(b'\n');
    out
}

fn format_shape(shape: &[u64]) -> String {
    match shape {
        [] => "()".to_string(),
        [d] => format!("({d},)"),
        dims => {
            let inner: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
            format!("({})", inner.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_64_aligned() {
        let bytes = build_header(&[12, 64, 8], NpyDtype::U8);
        assert_eq!(bytes.len() % ALIGNMENT, 0);
        assert!(bytes.starts_with(MAGIC));
        assert_eq!(bytes[6..8], VERSION);
    }

    #[test]
    fn encode_round_trip_uint8() {
        let data: Vec<u8> = (0..24).collect(); // shape (3,2,4) of uint8 = 24 bytes
        let blob = encode_npy(&[3, 2, 4], NpyDtype::U8, &data);
        // Header is multiple of 64; data is appended verbatim.
        assert!(blob.len() > data.len());
        assert_eq!(&blob[blob.len() - data.len()..], data.as_slice());
        // Header text contains the descr and shape we asked for.
        let header = std::str::from_utf8(&blob[10..blob.len() - data.len()]).unwrap();
        assert!(header.contains("'descr': '|u1'"));
        assert!(header.contains("'shape': (3, 2, 4)"));
        assert!(header.contains("'fortran_order': False"));
    }

    #[test]
    fn encode_uint16_endian() {
        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        let blob = encode_npy(&[2], NpyDtype::U16, &data);
        let header = std::str::from_utf8(&blob[10..blob.len() - data.len()]).unwrap();
        assert!(header.contains("'descr': '<u2'"));
        assert!(header.contains("'shape': (2,)"));
    }

    #[test]
    fn shape_zero_dim() {
        assert_eq!(format_shape(&[]), "()");
        assert_eq!(format_shape(&[5]), "(5,)");
        assert_eq!(format_shape(&[2, 3]), "(2, 3)");
    }
}
