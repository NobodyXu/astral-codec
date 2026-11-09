//! GNU sparse member maps.
//!
//! GNU tar stores a sparse file as its data regions followed by a map from
//! each stored region to its offset in the expanded file. Unmapped ranges
//! (holes) are implicitly zero.
//!
//! tar-framing supports three of the GNU sparse encodings:
//!
//! - The old GNU format (typeflag `'S'`), which stores up to four map entries
//!   in the member header and the rest in sparse extension blocks that follow
//!   it. The extension blocks are not counted in the header's `size`.
//! - pax sparse format 0.1, which stores the map in a `GNU.sparse.map` record.
//! - pax sparse format 1.0, which stores the map in decimal text at the start
//!   of the member payload, padded with NUL bytes to a block boundary.
//!
//! The pax sparse format 0.0 repeats `GNU.sparse.offset` and
//! `GNU.sparse.numbytes` records within one extended header, which is
//! incoherent with pax's last-record-wins semantics. It is rejected.
//!
//! See: <https://www.gnu.org/software/tar/manual/html_node/Sparse-Formats.html>

use std::{ops::Range, sync::Arc};

use crate::{
    BLOCK_SIZE, Block, FrameError, FrameErrorInner, PaxRecord, PaxValue,
    header::{NumericFormat, is_all_nul},
    pax::PaxRecords,
};

/// The pax vendor namespace used by GNU sparse records.
const GNU_VENDOR: &str = "GNU";
/// The prefix of every GNU sparse record name after the vendor namespace.
const SPARSE_PREFIX: &str = "sparse.";

/// The byte range of the four sparse map entries in an old GNU header.
const OLD_GNU_HEADER_ENTRIES_RANGE: Range<usize> = 386..482;
/// The byte index of the extension flag in an old GNU header.
const OLD_GNU_HEADER_IS_EXTENDED_OFFSET: usize = 482;
/// The byte range of the expanded file size in an old GNU header.
const OLD_GNU_REAL_SIZE_RANGE: Range<usize> = 483..495;
/// The byte range of the 21 sparse map entries in an old GNU extension block.
const OLD_GNU_EXTENSION_ENTRIES_RANGE: Range<usize> = 0..504;
/// The byte index of the extension flag in an old GNU extension block.
const OLD_GNU_EXTENSION_IS_EXTENDED_OFFSET: usize = 504;
/// The size of one encoded old GNU sparse map entry.
const OLD_GNU_ENTRY_SIZE: usize = 24;
/// The size of each numeric field in one old GNU sparse map entry.
const OLD_GNU_ENTRY_FIELD_SIZE: usize = 12;
/// The number of decimal digits in [`u64::MAX`].
const MAX_DECIMAL_DIGITS: usize = 20;

/// The encoding that supplied a [`SparseMap`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseFormat {
    /// The old GNU format, using typeflag `'S'`.
    OldGnu,
    /// pax sparse format 0.1, using a `GNU.sparse.map` record.
    Pax0_1,
    /// pax sparse format 1.0, using a map at the start of the member payload.
    Pax1_0,
}

/// One stored data region of a sparse member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseEntry {
    /// The offset of this region in the expanded file.
    pub offset: u64,
    /// The number of stored bytes in this region.
    pub length: u64,
}

/// A validated map from the stored payload of a sparse member to its expanded contents.
///
/// Entries are ordered by offset and do not overlap. Each entry lies within
/// [`Self::real_size`], and the entry lengths sum to [`Self::stored_size`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMap {
    format: SparseFormat,
    real_size: u64,
    stored_size: u64,
    entries: Vec<SparseEntry>,
}

pub(crate) type SharedSparseMap = Arc<SparseMap>;

impl SparseMap {
    fn new(
        position: u64,
        format: SparseFormat,
        real_size: u64,
        stored_size: u64,
        entries: Vec<SparseEntry>,
    ) -> Result<Self, FrameError> {
        let mut end = 0_u64;
        let mut total = 0_u64;
        for entry in &entries {
            if entry.offset < end {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "map entries are out of order or overlap",
                ));
            }
            end = entry
                .offset
                .checked_add(entry.length)
                .ok_or_else(|| FrameError::arithmetic_overflow(position, "sparse entry end"))?;
            if end > real_size {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "map entry extends beyond the real file size",
                ));
            }
            total = total
                .checked_add(entry.length)
                .ok_or_else(|| FrameError::arithmetic_overflow(position, "sparse stored size"))?;
        }
        if total != stored_size {
            return Err(FrameError::invalid_sparse_map(
                position,
                "map entry lengths do not match the stored data size",
            ));
        }
        Ok(Self {
            format,
            real_size,
            stored_size,
            entries,
        })
    }

    /// Returns the encoding that supplied this map.
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// Returns the size of the expanded file, including holes.
    pub fn real_size(&self) -> u64 {
        self.real_size
    }

    /// Returns the number of stored data bytes in the member payload.
    ///
    /// This excludes the map itself, even when the map is stored in the
    /// member payload.
    pub fn stored_size(&self) -> u64 {
        self.stored_size
    }

    /// Returns the stored data regions in payload order.
    pub fn entries(&self) -> &[SparseEntry] {
        &self.entries
    }
}

/// The sparse layout declared by a member header.
pub(crate) enum SparseHeader {
    /// The map is fully known from the header and its extensions.
    Complete(SparseMap),
    /// The map continues in the blocks after the header.
    Pending(PendingSparseMap),
}

/// An in-progress sparse map whose encoding continues after the member header.
#[derive(Debug)]
pub(crate) struct PendingSparseMap {
    real_size: u64,
    entries: Vec<SparseEntry>,
    /// The number of map bytes consumed after the member header.
    map_size: u64,
    encoding: PendingEncoding,
}

#[derive(Debug)]
enum PendingEncoding {
    /// Old GNU sparse extension blocks, which are not counted in the header size.
    OldGnu { stored_size: u64 },
    /// A pax 1.0 map at the start of the member payload.
    Pax1_0 {
        /// The remaining member payload, including any unread map blocks.
        remaining: u64,
        text: Pax1_0MapText,
    },
}

/// An incremental parser for the decimal text of a pax 1.0 map.
#[derive(Debug, Default)]
struct Pax1_0MapText {
    /// The number of map entries, once its line has been read.
    count: Option<u64>,
    /// An entry offset whose length has not been read yet.
    offset: Option<u64>,
    /// The digits of the current, unterminated line.
    digits: Vec<u8>,
    /// Whether the final entry has been read; only NUL padding may follow.
    complete: bool,
}

/// The result of consuming one sparse map block.
pub(crate) enum SparseProgress {
    /// More map blocks are required.
    Pending(PendingSparseMap),
    /// The map is complete. `remaining` member payload bytes follow.
    Complete { map: SparseMap, remaining: u64 },
}

impl SparseHeader {
    /// Parses the sparse map fields of an old GNU `'S'` header.
    pub(crate) fn old_gnu(
        position: u64,
        block: &Block,
        stored_size: u64,
    ) -> Result<Self, FrameError> {
        let real_size = NumericFormat::Gnu
            .parse(&block[OLD_GNU_REAL_SIZE_RANGE])
            .ok_or_else(|| FrameError::invalid_sparse_map(position, "invalid real file size"))?;
        let mut entries = Vec::new();
        let extended = parse_old_gnu_entries(
            position,
            &block[OLD_GNU_HEADER_ENTRIES_RANGE],
            block[OLD_GNU_HEADER_IS_EXTENDED_OFFSET],
            &mut entries,
        )?;
        if extended {
            return Ok(Self::Pending(PendingSparseMap {
                real_size,
                entries,
                map_size: 0,
                encoding: PendingEncoding::OldGnu { stored_size },
            }));
        }
        SparseMap::new(
            position,
            SparseFormat::OldGnu,
            real_size,
            stored_size,
            entries,
        )
        .map(Self::Complete)
    }

    /// Interprets the GNU sparse records in a member's local pax extension.
    ///
    /// Returns `None` when the extension contains no GNU sparse records.
    pub(crate) fn pax(
        position: u64,
        records: Option<&PaxRecords>,
        effective_size: u64,
    ) -> Result<Option<Self>, FrameError> {
        let mut sparse = PaxSparseRecords::default();
        for (name, value) in records
            .into_iter()
            .flat_map(PaxRecords::as_slice)
            .filter_map(sparse_record)
        {
            sparse.insert(position, name, value)?;
        }
        if !sparse.present {
            return Ok(None);
        }

        let unsupported = |version: String| {
            FrameError::at(
                position,
                FrameErrorInner::UnsupportedSparseFormat { version },
            )
        };
        let payload_map = match (sparse.major, sparse.minor) {
            (None, None) if sparse.offset_or_numbytes => {
                return Err(unsupported("0.0".to_owned()));
            }
            (Some(b"1"), Some(b"0")) => true,
            (Some(b"0"), Some(b"1")) | (None, None) => false,
            (Some(major), Some(minor)) => {
                return Err(unsupported(format!(
                    "{}.{}",
                    String::from_utf8_lossy(major),
                    String::from_utf8_lossy(minor)
                )));
            }
            _ => {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "only one of GNU.sparse.major and GNU.sparse.minor is present",
                ));
            }
        };
        if sparse.offset_or_numbytes {
            return Err(FrameError::invalid_sparse_map(
                position,
                "GNU.sparse.offset and GNU.sparse.numbytes are only valid in format 0.0",
            ));
        }

        // GNU tar treats `GNU.sparse.size` and `GNU.sparse.realsize` as
        // aliases, although it writes only the former for 0.1 and only the
        // latter for 1.0.
        let real_size = match (sparse.size, sparse.realsize) {
            (Some(size), Some(realsize)) if size != realsize => {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "GNU.sparse.size and GNU.sparse.realsize disagree",
                ));
            }
            (Some(size), _) | (None, Some(size)) => parse_decimal(position, size)?,
            (None, None) => {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "real file size is missing",
                ));
            }
        };

        if payload_map {
            if sparse.numblocks.is_some() || sparse.map.is_some() {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "format 1.0 stores its map in the member payload",
                ));
            }
            return Ok(Some(Self::Pending(PendingSparseMap {
                real_size,
                entries: Vec::new(),
                map_size: 0,
                encoding: PendingEncoding::Pax1_0 {
                    remaining: effective_size,
                    text: Pax1_0MapText::default(),
                },
            })));
        }

        let (Some(numblocks), Some(map)) = (sparse.numblocks, sparse.map) else {
            return Err(FrameError::invalid_sparse_map(
                position,
                "format 0.1 requires GNU.sparse.numblocks and GNU.sparse.map",
            ));
        };
        let numblocks = parse_decimal(position, numblocks)?;
        let entries = if map.is_empty() {
            Vec::new()
        } else {
            parse_decimal_pairs(position, map.split(|byte| *byte == b','))?
        };
        if u64::try_from(entries.len()).ok() != Some(numblocks) {
            return Err(FrameError::invalid_sparse_map(
                position,
                "GNU.sparse.numblocks does not match the number of map entries",
            ));
        }
        SparseMap::new(
            position,
            SparseFormat::Pax0_1,
            real_size,
            effective_size,
            entries,
        )
        .map(|map| Some(Self::Complete(map)))
    }
}

impl PendingSparseMap {
    /// Returns the number of meaningful bytes in the next map block.
    pub(crate) fn block_len(&self) -> usize {
        match &self.encoding {
            PendingEncoding::OldGnu { .. } => BLOCK_SIZE,
            PendingEncoding::Pax1_0 { remaining, .. } => {
                (*remaining).min(BLOCK_SIZE as u64) as usize
            }
        }
    }

    /// Returns the unread member payload size, if the map is stored in the payload.
    pub(crate) fn remaining_payload(&self) -> Option<u64> {
        match &self.encoding {
            PendingEncoding::OldGnu { .. } => None,
            PendingEncoding::Pax1_0 { remaining, .. } => Some(*remaining),
        }
    }

    /// Consumes one map block, rejecting a map larger than `limit` bytes.
    pub(crate) fn consume(
        mut self,
        header_position: u64,
        block: &Block,
        limit: u64,
    ) -> Result<SparseProgress, FrameError> {
        self.map_size = self
            .map_size
            .checked_add(BLOCK_SIZE as u64)
            .ok_or_else(|| FrameError::arithmetic_overflow(header_position, "sparse map size"))?;
        if self.map_size > limit {
            return Err(FrameError::at(
                header_position,
                FrameErrorInner::SparseMapTooLarge { limit },
            ));
        }
        let len = self.block_len();
        match &mut self.encoding {
            PendingEncoding::OldGnu { stored_size } => {
                let stored_size = *stored_size;
                if !is_all_nul(&block[OLD_GNU_EXTENSION_IS_EXTENDED_OFFSET + 1..]) {
                    return Err(FrameError::invalid_sparse_map(
                        header_position,
                        "extension block padding is not NUL",
                    ));
                }
                let extended = parse_old_gnu_entries(
                    header_position,
                    &block[OLD_GNU_EXTENSION_ENTRIES_RANGE],
                    block[OLD_GNU_EXTENSION_IS_EXTENDED_OFFSET],
                    &mut self.entries,
                )?;
                if extended {
                    return Ok(SparseProgress::Pending(self));
                }
                self.complete(header_position, SparseFormat::OldGnu, stored_size)
            }
            PendingEncoding::Pax1_0 { remaining, text } => {
                *remaining -= len as u64;
                let remaining = *remaining;
                text.parse(header_position, &block[..len], &mut self.entries)?;
                if text.complete {
                    return self.complete(header_position, SparseFormat::Pax1_0, remaining);
                }
                if remaining == 0 {
                    return Err(FrameError::invalid_sparse_map(
                        header_position,
                        "map extends beyond the member payload",
                    ));
                }
                Ok(SparseProgress::Pending(self))
            }
        }
    }

    fn complete(
        self,
        position: u64,
        format: SparseFormat,
        stored_size: u64,
    ) -> Result<SparseProgress, FrameError> {
        let map = SparseMap::new(position, format, self.real_size, stored_size, self.entries)?;
        Ok(SparseProgress::Complete {
            map,
            remaining: stored_size,
        })
    }
}

impl Pax1_0MapText {
    /// Parses one block of map text, appending each completed entry to `entries`.
    fn parse(
        &mut self,
        position: u64,
        bytes: &[u8],
        entries: &mut Vec<SparseEntry>,
    ) -> Result<(), FrameError> {
        for byte in bytes {
            match *byte {
                0 if self.complete => {}
                _ if self.complete => {
                    return Err(FrameError::invalid_sparse_map(
                        position,
                        "map padding is not NUL",
                    ));
                }
                b'0'..=b'9' if self.digits.len() < MAX_DECIMAL_DIGITS => {
                    self.digits.push(*byte);
                }
                b'\n' => {
                    let number = parse_decimal(position, &self.digits)?;
                    self.digits.clear();
                    self.accept(position, number, entries)?;
                }
                _ => {
                    return Err(FrameError::invalid_sparse_map(
                        position,
                        "invalid decimal value",
                    ));
                }
            }
        }
        Ok(())
    }

    fn accept(
        &mut self,
        position: u64,
        number: u64,
        entries: &mut Vec<SparseEntry>,
    ) -> Result<(), FrameError> {
        let Some(count) = self.count else {
            self.count = Some(number);
            self.complete = number == 0;
            return Ok(());
        };
        let Some(offset) = self.offset.take() else {
            self.offset = Some(number);
            return Ok(());
        };
        entries.push(SparseEntry {
            offset,
            length: number,
        });
        let read = u64::try_from(entries.len())
            .map_err(|_| FrameError::arithmetic_overflow(position, "sparse map entry count"))?;
        self.complete = read == count;
        Ok(())
    }
}

/// The GNU sparse records of one local pax extension.
#[derive(Default)]
struct PaxSparseRecords<'a> {
    present: bool,
    major: Option<&'a [u8]>,
    minor: Option<&'a [u8]>,
    size: Option<&'a [u8]>,
    realsize: Option<&'a [u8]>,
    numblocks: Option<&'a [u8]>,
    map: Option<&'a [u8]>,
    offset_or_numbytes: bool,
}

impl<'a> PaxSparseRecords<'a> {
    fn insert(&mut self, position: u64, name: &str, value: &'a [u8]) -> Result<(), FrameError> {
        self.present = true;
        let field = match name {
            "major" => &mut self.major,
            "minor" => &mut self.minor,
            "size" => &mut self.size,
            "realsize" => &mut self.realsize,
            "numblocks" => &mut self.numblocks,
            "map" => &mut self.map,
            // The logical layer resolves the member path.
            "name" => return Ok(()),
            "offset" | "numbytes" => {
                self.offset_or_numbytes = true;
                return Ok(());
            }
            _ => {
                return Err(FrameError::invalid_sparse_map(
                    position,
                    "unknown GNU sparse record",
                ));
            }
        };
        // As with every other pax record, the last record takes effect.
        *field = Some(value);
        Ok(())
    }
}

/// Returns the name suffix and value of a `GNU.sparse.*` pax record.
///
/// A deleted record is returned with an empty value, which every numeric
/// sparse field rejects.
pub(crate) fn sparse_record(record: &PaxRecord) -> Option<(&str, &[u8])> {
    if let PaxRecord::Vendor {
        vendor,
        name,
        value,
    } = record
        && vendor.as_ref() == GNU_VENDOR
        && let Some(name) = name.strip_prefix(SPARSE_PREFIX)
    {
        let value = match value {
            PaxValue::Value(value) => value.as_ref(),
            PaxValue::Deleted => &[],
        };
        return Some((name, value));
    }
    None
}

/// Parses consecutive old GNU sparse map entries, returning whether the map is extended.
///
/// Entries end at the first wholly NUL entry. Every later entry must also be
/// wholly NUL, and a map with an empty entry cannot be extended. GNU tar stops
/// reading at the first empty entry while other readers continue to the next
/// extension block, so accepting either combination would be ambiguous.
fn parse_old_gnu_entries(
    position: u64,
    bytes: &[u8],
    is_extended: u8,
    entries: &mut Vec<SparseEntry>,
) -> Result<bool, FrameError> {
    let extended = match is_extended {
        0 => false,
        1 => true,
        _ => {
            return Err(FrameError::invalid_sparse_map(
                position,
                "invalid extension flag",
            ));
        }
    };
    let mut ended = false;
    for entry in bytes.chunks_exact(OLD_GNU_ENTRY_SIZE) {
        if is_all_nul(entry) {
            ended = true;
        } else if ended {
            return Err(FrameError::invalid_sparse_map(
                position,
                "map entry follows an empty entry",
            ));
        } else {
            let (offset, length) = entry.split_at(OLD_GNU_ENTRY_FIELD_SIZE);
            let parse = |field| {
                NumericFormat::Gnu
                    .parse(field)
                    .ok_or_else(|| FrameError::invalid_sparse_map(position, "invalid map entry"))
            };
            entries.push(SparseEntry {
                offset: parse(offset)?,
                length: parse(length)?,
            });
        }
    }
    if ended && extended {
        return Err(FrameError::invalid_sparse_map(
            position,
            "extended map contains an empty entry",
        ));
    }
    Ok(extended)
}

fn parse_decimal_pairs<'a>(
    position: u64,
    numbers: impl Iterator<Item = &'a [u8]>,
) -> Result<Vec<SparseEntry>, FrameError> {
    let mut entries = Vec::new();
    let mut offset = None;
    for number in numbers {
        let number = parse_decimal(position, number)?;
        match offset.take() {
            Some(offset) => entries.push(SparseEntry {
                offset,
                length: number,
            }),
            None => offset = Some(number),
        }
    }
    if offset.is_some() {
        return Err(FrameError::invalid_sparse_map(
            position,
            "map has an offset without a length",
        ));
    }
    Ok(entries)
}

/// Parses a strict decimal sparse map number.
///
/// Like pax `size` records, multi-digit values cannot start with zero, since
/// some readers interpret such values as octal.
fn parse_decimal(position: u64, value: &[u8]) -> Result<u64, FrameError> {
    if value.is_empty() || value.starts_with(b"0") && value.len() > 1 {
        return Err(FrameError::invalid_sparse_map(
            position,
            "invalid decimal value",
        ));
    }
    std::str::from_utf8(value)
        .ok()
        .filter(|value| value.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| FrameError::invalid_sparse_map(position, "invalid decimal value"))
}
