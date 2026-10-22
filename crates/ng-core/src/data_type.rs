//! Voxel data types. Port of `src/util/data_type.ts`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Float32,
}

impl DataType {
    pub const ALL: [DataType; 8] = [
        DataType::Uint8,
        DataType::Int8,
        DataType::Uint16,
        DataType::Int16,
        DataType::Uint32,
        DataType::Int32,
        DataType::Uint64,
        DataType::Float32,
    ];

    /// `DATA_TYPE_BYTES`.
    pub fn bytes(self) -> usize {
        match self {
            DataType::Uint8 | DataType::Int8 => 1,
            DataType::Uint16 | DataType::Int16 => 2,
            DataType::Uint32 | DataType::Int32 | DataType::Float32 => 4,
            DataType::Uint64 => 8,
        }
    }

    /// `DATA_TYPE_SIGNED`: `None` for float.
    pub fn signed(self) -> Option<bool> {
        match self {
            DataType::Int8 | DataType::Int16 | DataType::Int32 => Some(true),
            DataType::Float32 => None,
            _ => Some(false),
        }
    }

    /// The enum member's name, as neuroglancer writes it (`"UINT8"`, …).
    pub fn name(self) -> &'static str {
        match self {
            DataType::Uint8 => "UINT8",
            DataType::Int8 => "INT8",
            DataType::Uint16 => "UINT16",
            DataType::Int16 => "INT16",
            DataType::Uint32 => "UINT32",
            DataType::Int32 => "INT32",
            DataType::Uint64 => "UINT64",
            DataType::Float32 => "FLOAT32",
        }
    }

    /// Case-insensitive, as neuroglancer's datasources accept `"uint8"` and
    /// `"UINT8"` alike.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for t in DataType::ALL {
            assert_eq!(DataType::from_name(t.name()), Some(t));
            assert_eq!(DataType::from_name(&t.name().to_lowercase()), Some(t));
        }
        assert_eq!(DataType::from_name("float64"), None);
    }
}
