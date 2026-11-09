pub mod support;

use std::{
    error::Error,
    io,
    pin::{Pin, pin},
    sync::Arc,
    task::{Context, Poll, Waker},
};

use support::{ArchiveBuilder, ArchiveFormat, header, pax_record, raw_pax_record, set_checksum};
use tar_codec::{
    Archive as _, DecodeError, DecodePolicy, DecodePolicyViolation, Member, MemberPayload,
    TarArchive, extract::ExtractPolicy,
};
use tar_framing::{
    BLOCK_SIZE, Block, FrameError, FrameErrorInner, PaxKeyword, StreamPolicy,
    header::LINK_NAME_RANGE,
    logical::TarReader,
    sparse::{SparseEntry, SparseFormat},
    stream::DataOwner,
};
use tempfile::tempdir;
use tokio::io::{AsyncRead, ReadBuf};

type TestResult = Result<(), Box<dyn Error>>;

/// The byte range of the four sparse map entries in an old GNU header.
const OLD_GNU_ENTRIES_START: usize = 386;
/// The byte index of the extension flag in an old GNU header.
const OLD_GNU_IS_EXTENDED_OFFSET: usize = 482;
/// The byte offset of the expanded file size in an old GNU header.
const OLD_GNU_REAL_SIZE_START: usize = 483;
/// The byte index of the extension flag in an old GNU sparse extension block.
const EXTENSION_IS_EXTENDED_OFFSET: usize = 504;

fn sparse_policy() -> DecodePolicy {
    DecodePolicy::default().allow_gnu_sparse(true)
}

async fn read_payload<P: MemberPayload<Error = DecodeError>>(
    mut payload: P,
) -> Result<Vec<u8>, DecodeError> {
    let mut data = Vec::new();
    let mut chunk = Vec::new();
    while payload.next_chunk(&mut chunk, 3).await? {
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

/// A sparse file and the regions that a writer would store.
struct SparseFile {
    contents: Vec<u8>,
    entries: Vec<SparseEntry>,
}

impl SparseFile {
    /// Builds a file of `real_size` bytes whose stored regions contain a byte pattern.
    fn new(real_size: u64, regions: &[(u64, u64)]) -> Self {
        let mut contents = vec![0; real_size as usize];
        let entries = regions
            .iter()
            .map(|&(offset, length)| {
                for (index, byte) in contents[offset as usize..(offset + length) as usize]
                    .iter_mut()
                    .enumerate()
                {
                    *byte = (offset as usize + index) as u8 | 1;
                }
                SparseEntry { offset, length }
            })
            .collect();
        Self { contents, entries }
    }

    /// Returns a layout with leading, interior, and trailing holes, as GNU tar
    /// writes it: the final entry is an empty region at the end of the file.
    fn with_holes() -> Self {
        Self::new(
            20_000,
            &[(4_096, 600), (9_000, 1), (12_288, 4_096), (20_000, 0)],
        )
    }

    /// Returns a layout whose map needs several old GNU extension blocks and
    /// several pax 1.0 map blocks.
    fn fragmented() -> Self {
        let regions = (0..60)
            .map(|index| (index * 1_000 + 500, 10))
            .collect::<Vec<_>>();
        Self::new(60_000, &regions)
    }

    fn real_size(&self) -> u64 {
        self.contents.len() as u64
    }

    fn stored(&self) -> Vec<u8> {
        self.entries
            .iter()
            .flat_map(|entry| {
                &self.contents[entry.offset as usize..(entry.offset + entry.length) as usize]
            })
            .copied()
            .collect()
    }

    fn pax_0_1_records(&self, name: &str) -> Vec<u8> {
        let map = self
            .entries
            .iter()
            .flat_map(|entry| [entry.offset.to_string(), entry.length.to_string()])
            .collect::<Vec<_>>()
            .join(",");
        [
            sparse_record("size", &self.real_size().to_string()),
            sparse_record("numblocks", &self.entries.len().to_string()),
            sparse_record("name", name),
            sparse_record("map", &map),
        ]
        .concat()
    }

    fn pax_1_0_records(&self, name: &str) -> Vec<u8> {
        [
            sparse_record("major", "1"),
            sparse_record("minor", "0"),
            sparse_record("name", name),
            sparse_record("realsize", &self.real_size().to_string()),
        ]
        .concat()
    }

    /// Returns the pax 1.0 member payload: the block-padded map, then the stored data.
    fn pax_1_0_payload(&self) -> Vec<u8> {
        let mut payload = format!("{}\n", self.entries.len()).into_bytes();
        for entry in &self.entries {
            payload.extend_from_slice(format!("{}\n{}\n", entry.offset, entry.length).as_bytes());
        }
        payload.resize(payload.len().next_multiple_of(BLOCK_SIZE), 0);
        payload.extend_from_slice(&self.stored());
        payload
    }

    /// Appends an old GNU sparse member, with extension blocks for entries
    /// beyond the four that fit in its header.
    fn append_old_gnu(&self, archive: &mut ArchiveBuilder, name: &str) {
        let (header_entries, rest) = self.entries.split_at(self.entries.len().min(4));
        let extensions = rest.chunks(21).collect::<Vec<_>>();
        archive.block(&old_gnu_header(
            name,
            self.stored().len() as u64,
            self.real_size(),
            header_entries,
            !extensions.is_empty(),
        ));
        for (index, entries) in extensions.iter().enumerate() {
            archive.block(&old_gnu_extension(entries, index + 1 < extensions.len()));
        }
        append_payload(archive, &self.stored());
    }

    fn append(&self, archive: &mut ArchiveBuilder, format: SparseFormat, name: &str) {
        match format {
            SparseFormat::OldGnu => self.append_old_gnu(archive, name),
            SparseFormat::Pax0_1 => {
                archive.pax(b'x', &self.pax_0_1_records(name)).ustar(
                    "GNUSparseFile.1/file",
                    b'0',
                    &self.stored(),
                    "",
                    0o755,
                );
            }
            SparseFormat::Pax1_0 => {
                archive.pax(b'x', &self.pax_1_0_records(name)).ustar(
                    "GNUSparseFile.1/file",
                    b'0',
                    &self.pax_1_0_payload(),
                    "",
                    0o755,
                );
            }
        }
    }
}

/// Appends a regular file in the archive family used by `format`.
fn append_regular(archive: &mut ArchiveBuilder, format: SparseFormat, name: &str, payload: &[u8]) {
    match format {
        SparseFormat::OldGnu => archive.gnu(name, b'0', payload, "", 0o644),
        SparseFormat::Pax0_1 | SparseFormat::Pax1_0 => {
            archive.ustar(name, b'0', payload, "", 0o644)
        }
    };
}

fn sparse_record(name: &str, value: &str) -> Vec<u8> {
    pax_record(
        PaxKeyword::Vendor {
            vendor: Arc::from("GNU"),
            name: Arc::from(format!("sparse.{name}")),
        },
        value,
    )
}

fn append_payload(archive: &mut ArchiveBuilder, payload: &[u8]) {
    for chunk in payload.chunks(BLOCK_SIZE) {
        let mut block = [0; BLOCK_SIZE];
        block[..chunk.len()].copy_from_slice(chunk);
        archive.block(&block);
    }
}

fn set_octal(field: &mut [u8], value: u64) {
    let width = field.len() - 1;
    field.copy_from_slice(format!("{value:0width$o}\0").as_bytes());
}

fn set_entries(entries_field: &mut [u8], entries: &[SparseEntry]) {
    for (field, entry) in entries_field.chunks_exact_mut(24).zip(entries) {
        set_octal(&mut field[..12], entry.offset);
        set_octal(&mut field[12..], entry.length);
    }
}

fn old_gnu_header(
    name: &str,
    stored_size: u64,
    real_size: u64,
    entries: &[SparseEntry],
    extended: bool,
) -> Block {
    let mut block = header(ArchiveFormat::Gnu, name, b'S', stored_size, "", 0o755);
    set_entries(
        &mut block[OLD_GNU_ENTRIES_START..OLD_GNU_IS_EXTENDED_OFFSET],
        entries,
    );
    block[OLD_GNU_IS_EXTENDED_OFFSET] = u8::from(extended);
    set_octal(
        &mut block[OLD_GNU_REAL_SIZE_START..OLD_GNU_REAL_SIZE_START + 12],
        real_size,
    );
    set_checksum(&mut block);
    block
}

fn old_gnu_extension(entries: &[SparseEntry], extended: bool) -> Block {
    let mut block = [0; BLOCK_SIZE];
    set_entries(&mut block[..EXTENSION_IS_EXTENDED_OFFSET], entries);
    block[EXTENSION_IS_EXTENDED_OFFSET] = u8::from(extended);
    block
}

const FORMATS: [SparseFormat; 3] = [
    SparseFormat::OldGnu,
    SparseFormat::Pax0_1,
    SparseFormat::Pax1_0,
];

/// Decodes the first member of `bytes`, discarding it.
async fn decode_first(bytes: &[u8], policy: DecodePolicy) -> Result<(), DecodeError> {
    TarArchive::new(bytes)
        .with_policy(policy)
        .members()
        .next()
        .await
        .map(|_| ())
}

fn sparse_error(result: Result<(), DecodeError>) -> Option<&'static str> {
    match result {
        Err(DecodeError::Framing(FrameError {
            inner: FrameErrorInner::InvalidSparseMap { reason },
            ..
        })) => Some(reason),
        _ => None,
    }
}

#[tokio::test]
async fn decodes_sparse_files_into_their_expanded_contents() -> TestResult {
    for file in [
        SparseFile::with_holes(),
        SparseFile::fragmented(),
        SparseFile::new(0, &[]),
        SparseFile::new(5_000, &[]),
        SparseFile::new(700, &[(0, 700)]),
    ] {
        for format in FORMATS {
            let mut archive = ArchiveBuilder::new();
            file.append(&mut archive, format, "sparse");
            append_regular(&mut archive, format, "after", b"after");
            let bytes = archive.finish();

            let mut members = TarArchive::new(bytes.as_slice())
                .with_policy(sparse_policy())
                .members();
            let Some(Member::File {
                metadata,
                size,
                executable,
                payload,
            }) = members.next().await?
            else {
                return Err(io::Error::other("expected sparse file member").into());
            };
            assert_eq!(metadata.path, "sparse", "{format:?}");
            assert_eq!(size, file.real_size(), "{format:?}");
            assert!(executable, "{format:?}");
            assert!(
                read_payload(payload).await? == file.contents,
                "{format:?} expanded contents differ"
            );

            let Some(Member::File {
                metadata, payload, ..
            }) = members.next().await?
            else {
                return Err(io::Error::other("expected member after sparse file").into());
            };
            assert_eq!(metadata.path, "after", "{format:?}");
            assert_eq!(read_payload(payload).await?, b"after");
            assert!(members.next().await?.is_none());
        }
    }
    Ok(())
}

/// A reader that returns [`Poll::Pending`] before every short read.
struct InterruptingReader {
    bytes: Vec<u8>,
    position: usize,
    pending: bool,
}

impl AsyncRead for InterruptingReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.pending = !self.pending;
        if self.pending {
            context.waker().wake_by_ref();
            return Poll::Pending;
        }
        let end = self
            .bytes
            .len()
            .min(self.position + 100)
            .min(self.position + buffer.remaining());
        buffer.put_slice(&self.bytes[self.position..end]);
        self.position = end;
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn cancelled_sparse_reads_can_be_retried() -> TestResult {
    let file = SparseFile::fragmented();
    for format in FORMATS {
        let mut archive = ArchiveBuilder::new();
        file.append(&mut archive, format, "sparse");
        let reader = InterruptingReader {
            bytes: archive.finish(),
            position: 0,
            pending: false,
        };
        let mut archive = TarArchive::new(reader).with_policy(sparse_policy());
        let mut context = Context::from_waker(Waker::noop());
        let member = loop {
            // Each pending poll drops, and therefore cancels, the read.
            if let Poll::Ready(member) = pin!(archive.next_member()).poll(&mut context) {
                break member?;
            }
        };
        let Some(Member::File { mut payload, .. }) = member else {
            return Err(io::Error::other("expected sparse file member").into());
        };
        let mut contents = Vec::new();
        let mut chunk = Vec::new();
        loop {
            let poll = pin!(payload.next_chunk(&mut chunk, 700)).poll(&mut context);
            match poll {
                Poll::Ready(Ok(true)) => contents.extend_from_slice(&chunk),
                Poll::Ready(Ok(false)) => break,
                Poll::Ready(Err(error)) => return Err(error.into()),
                Poll::Pending => {}
            }
        }
        assert!(contents == file.contents, "{format:?}");
    }
    Ok(())
}

#[tokio::test]
async fn skips_unread_sparse_payloads() -> TestResult {
    for format in FORMATS {
        let mut archive = ArchiveBuilder::new();
        SparseFile::fragmented().append(&mut archive, format, "sparse");
        append_regular(&mut archive, format, "after", b"");
        let bytes = archive.finish();

        let mut members = TarArchive::new(bytes.as_slice())
            .with_policy(sparse_policy())
            .members();
        assert!(matches!(members.next().await?, Some(Member::File { .. })));
        assert!(matches!(
            members.next().await?,
            Some(Member::File { metadata, .. }) if metadata.path == "after"
        ));
    }
    Ok(())
}

#[tokio::test]
async fn exposes_sparse_maps_on_logical_members() -> TestResult {
    let file = SparseFile::fragmented();
    for format in FORMATS {
        let mut archive = ArchiveBuilder::new();
        file.append(&mut archive, format, "sparse");
        let bytes = archive.finish();

        let mut reader = TarReader::new(bytes.as_slice())
            .with_policy(StreamPolicy::default().allow_gnu_sparse(true));
        let mut member = reader.next_frame().await?.expect("expected sparse member");
        let map = member.header.sparse_map.expect("expected sparse map");
        assert_eq!(map.format(), format);
        assert_eq!(map.real_size(), file.real_size());
        assert_eq!(map.entries(), file.entries);
        assert_eq!(member.header.effective_size, map.stored_size());

        // The logical payload is the stored data, without any map blocks.
        let mut stored = Vec::new();
        while let Some(block) = member.payload.next_block().await? {
            stored.extend_from_slice(&block.block[..block.len]);
        }
        assert_eq!(stored, file.stored());
    }
    Ok(())
}

#[tokio::test]
async fn extracts_sparse_files() -> TestResult {
    let file = SparseFile::with_holes();
    for format in FORMATS {
        let mut archive = ArchiveBuilder::new();
        file.append(&mut archive, format, "dir/sparse");
        let bytes = archive.finish();

        let temp = tempdir()?;
        TarArchive::new(bytes.as_slice())
            .with_policy(sparse_policy())
            .extract_in(temp.path(), ExtractPolicy::default())
            .await?;
        assert!(std::fs::read(temp.path().join("dir/sparse"))? == file.contents);
    }
    Ok(())
}

#[tokio::test]
async fn sparse_members_require_opt_in() -> TestResult {
    for format in FORMATS {
        let mut archive = ArchiveBuilder::new();
        SparseFile::with_holes().append(&mut archive, format, "sparse");
        let bytes = archive.finish();

        let result = decode_first(&bytes, DecodePolicy::default()).await;
        match format {
            SparseFormat::OldGnu => assert!(matches!(
                result,
                Err(DecodeError::Framing(FrameError {
                    inner: FrameErrorInner::UnsupportedTypeflag { typeflag: b'S' },
                    ..
                }))
            )),
            _ => assert!(matches!(
                result,
                Err(DecodeError::PolicyViolation {
                    violation: DecodePolicyViolation::PaxVendorExtension { vendor, name },
                    ..
                }) if vendor == "GNU" && name.starts_with("sparse.")
            )),
        }
    }
    Ok(())
}

#[tokio::test]
async fn pax_sparse_names_take_precedence_over_paths() -> TestResult {
    let file = SparseFile::with_holes();
    let path = pax_record(PaxKeyword::Path, "GNUSparseFile.1/a-very-long-name");
    for (records, payload, expected) in [
        (
            [path.clone(), file.pax_0_1_records("real")].concat(),
            file.stored(),
            "real",
        ),
        (
            [path.clone(), file.pax_1_0_records("real")].concat(),
            file.pax_1_0_payload(),
            "real",
        ),
        // Without `GNU.sparse.name`, the ordinary path rules apply.
        (
            [
                path.clone(),
                sparse_record("major", "1"),
                sparse_record("minor", "0"),
                sparse_record("realsize", "20000"),
            ]
            .concat(),
            file.pax_1_0_payload(),
            "GNUSparseFile.1/a-very-long-name",
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &records)
            .ustar("GNUSparseFile.1/a", b'0', &payload, "", 0o644);
        let bytes = archive.finish();

        let mut members = TarArchive::new(bytes.as_slice())
            .with_policy(sparse_policy())
            .members();
        let Some(Member::File {
            metadata, payload, ..
        }) = members.next().await?
        else {
            return Err(io::Error::other("expected sparse file member").into());
        };
        assert_eq!(metadata.path, expected);
        assert!(read_payload(payload).await? == file.contents);
    }

    // A deleted sparse name cannot fall back to the renamed ustar path.
    let mut archive = ArchiveBuilder::new();
    archive.pax(b'x', &file.pax_0_1_records("")).ustar(
        "GNUSparseFile.1/a",
        b'0',
        &file.stored(),
        "",
        0o644,
    );
    let bytes = archive.finish();
    let result = decode_first(&bytes, sparse_policy()).await;
    assert!(matches!(
        result,
        Err(DecodeError::Framing(FrameError {
            inner: FrameErrorInner::DeletedPaxRecord {
                keyword: "GNU.sparse.name"
            },
            ..
        }))
    ));
    Ok(())
}

#[tokio::test]
async fn sparse_records_remain_subject_to_other_pax_policy() -> TestResult {
    let file = SparseFile::with_holes();
    for records in [
        // Unknown vendor records on a sparse member are still rejected.
        [file.pax_0_1_records("sparse"), gnu_vendor_record()].concat(),
        // Repeated sparse records are still duplicates.
        [
            file.pax_0_1_records("sparse"),
            sparse_record("name", "other"),
        ]
        .concat(),
    ] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &records)
            .ustar("GNUSparseFile.1/a", b'0', &file.stored(), "", 0o644);
        let bytes = archive.finish();
        assert!(matches!(
            decode_first(&bytes, sparse_policy()).await,
            Err(DecodeError::PolicyViolation { .. })
        ));
    }
    Ok(())
}

fn gnu_vendor_record() -> Vec<u8> {
    raw_pax_record(
        PaxKeyword::Vendor {
            vendor: Arc::from("GNU"),
            name: Arc::from("dumpdir"),
        },
        b"value",
    )
}

#[tokio::test]
async fn rejects_malformed_old_gnu_sparse_maps() -> TestResult {
    let entry = |offset, length| SparseEntry { offset, length };
    let entries = [entry(0, 10), entry(100, 10)];
    let mut extension_padding = old_gnu_extension(&entries, false);
    extension_padding[BLOCK_SIZE - 1] = 1;
    let mut empty_then_entry = old_gnu_header("sparse", 20, 200, &entries, false);
    set_entries(
        &mut empty_then_entry[OLD_GNU_ENTRIES_START + 72..OLD_GNU_IS_EXTENDED_OFFSET],
        &[entry(150, 0)],
    );
    set_checksum(&mut empty_then_entry);
    let mut invalid_flag = old_gnu_header("sparse", 20, 200, &entries, false);
    invalid_flag[OLD_GNU_IS_EXTENDED_OFFSET] = 2;
    set_checksum(&mut invalid_flag);
    let mut invalid_entry = old_gnu_header("sparse", 20, 200, &entries, false);
    invalid_entry[OLD_GNU_ENTRIES_START] = b'9';
    set_checksum(&mut invalid_entry);

    for (case, blocks, expected) in [
        (
            "out of order",
            vec![old_gnu_header(
                "sparse",
                20,
                200,
                &[entry(100, 10), entry(0, 10)],
                false,
            )],
            "map entries are out of order or overlap",
        ),
        (
            "overlapping",
            vec![old_gnu_header(
                "sparse",
                20,
                200,
                &[entry(0, 10), entry(5, 10)],
                false,
            )],
            "map entries are out of order or overlap",
        ),
        (
            "beyond real size",
            vec![old_gnu_header("sparse", 20, 105, &entries, false)],
            "map entry extends beyond the real file size",
        ),
        (
            "stored size mismatch",
            vec![old_gnu_header("sparse", 21, 200, &entries, false)],
            "map entry lengths do not match the stored data size",
        ),
        (
            "empty entry before an extension",
            vec![
                old_gnu_header("sparse", 20, 200, &entries, true),
                old_gnu_extension(&[], false),
            ],
            "extended map contains an empty entry",
        ),
        (
            "entry after an empty entry",
            vec![empty_then_entry],
            "map entry follows an empty entry",
        ),
        (
            "invalid extension flag",
            vec![invalid_flag],
            "invalid extension flag",
        ),
        ("invalid entry", vec![invalid_entry], "invalid map entry"),
        (
            "extension padding",
            vec![
                old_gnu_header(
                    "sparse",
                    20,
                    200,
                    &[entry(0, 0), entry(0, 0), entry(0, 0), entry(0, 0)],
                    true,
                ),
                extension_padding,
            ],
            "extension block padding is not NUL",
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        for block in &blocks {
            archive.block(block);
        }
        append_payload(&mut archive, &[1; 20]);
        let bytes = archive.finish();
        let result = decode_first(&bytes, sparse_policy()).await;
        assert_eq!(sparse_error(result), Some(expected), "{case}");
    }
    Ok(())
}

#[tokio::test]
async fn old_gnu_sparse_members_cannot_have_link_targets() -> TestResult {
    let mut block = old_gnu_header("sparse", 0, 10, &[], false);
    block[LINK_NAME_RANGE.start] = b't';
    set_checksum(&mut block);
    let mut archive = ArchiveBuilder::new();
    archive.block(&block);
    let bytes = archive.finish();
    assert!(matches!(
        decode_first(&bytes, sparse_policy()).await,
        Err(DecodeError::Framing(FrameError {
            inner: FrameErrorInner::UnexpectedLinkTarget { .. },
            ..
        }))
    ));
    Ok(())
}

#[tokio::test]
async fn rejects_malformed_pax_sparse_records() -> TestResult {
    let size = sparse_record("size", "200");
    let numblocks = sparse_record("numblocks", "2");
    let map = sparse_record("map", "0,10,100,10");
    for (case, records, expected) in [
        (
            "numblocks mismatch",
            [size.clone(), sparse_record("numblocks", "3"), map.clone()].concat(),
            "GNU.sparse.numblocks does not match the number of map entries",
        ),
        (
            "odd map",
            [
                size.clone(),
                numblocks.clone(),
                sparse_record("map", "0,10,100"),
            ]
            .concat(),
            "map has an offset without a length",
        ),
        (
            "missing numblocks",
            [size.clone(), map.clone()].concat(),
            "format 0.1 requires GNU.sparse.numblocks and GNU.sparse.map",
        ),
        (
            "missing map",
            [size.clone(), numblocks.clone()].concat(),
            "format 0.1 requires GNU.sparse.numblocks and GNU.sparse.map",
        ),
        (
            "missing size",
            [numblocks.clone(), map.clone()].concat(),
            "real file size is missing",
        ),
        (
            "disagreeing sizes",
            [
                size.clone(),
                sparse_record("realsize", "300"),
                numblocks.clone(),
                map.clone(),
            ]
            .concat(),
            "GNU.sparse.size and GNU.sparse.realsize disagree",
        ),
        (
            "leading zero",
            [
                sparse_record("size", "0200"),
                numblocks.clone(),
                map.clone(),
            ]
            .concat(),
            "invalid decimal value",
        ),
        (
            "signed value",
            [
                size.clone(),
                numblocks.clone(),
                sparse_record("map", "0,10,+100,10"),
            ]
            .concat(),
            "invalid decimal value",
        ),
        (
            "unknown record",
            [
                size.clone(),
                numblocks.clone(),
                map.clone(),
                sparse_record("other", "1"),
            ]
            .concat(),
            "unknown GNU sparse record",
        ),
        (
            "version without minor",
            [sparse_record("major", "1"), size.clone()].concat(),
            "only one of GNU.sparse.major and GNU.sparse.minor is present",
        ),
        (
            "1.0 with a record map",
            [
                sparse_record("major", "1"),
                sparse_record("minor", "0"),
                size.clone(),
                map.clone(),
            ]
            .concat(),
            "format 1.0 stores its map in the member payload",
        ),
        (
            "1.0 with 0.0 records",
            [
                sparse_record("major", "1"),
                sparse_record("minor", "0"),
                size.clone(),
                sparse_record("offset", "0"),
            ]
            .concat(),
            "GNU.sparse.offset and GNU.sparse.numbytes are only valid in format 0.0",
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &records)
            .ustar("sparse", b'0', &[1; 20], "", 0o644);
        let bytes = archive.finish();
        let result = decode_first(&bytes, sparse_policy()).await;
        assert_eq!(sparse_error(result), Some(expected), "{case}");
    }
    Ok(())
}

#[tokio::test]
async fn rejects_sparse_records_on_other_member_types() -> TestResult {
    let file = SparseFile::new(10, &[]);
    for (typeflag, link_name) in [(b'7', ""), (b'2', "target"), (b'5', "")] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &file.pax_0_1_records("sparse"))
            .ustar("member", typeflag, b"", link_name, 0o644);
        let bytes = archive.finish();
        let result = decode_first(&bytes, sparse_policy()).await;
        assert_eq!(
            sparse_error(result),
            Some("sparse member is not a regular file"),
            "{typeflag}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn rejects_unsupported_pax_sparse_versions() -> TestResult {
    for (records, expected) in [
        // Format 0.0 has no version records and repeats offset and numbytes.
        (
            [
                sparse_record("size", "200"),
                sparse_record("numblocks", "1"),
                sparse_record("offset", "0"),
                sparse_record("numbytes", "20"),
            ]
            .concat(),
            "0.0",
        ),
        (
            [
                sparse_record("major", "2"),
                sparse_record("minor", "0"),
                sparse_record("realsize", "200"),
            ]
            .concat(),
            "2.0",
        ),
        (
            [
                sparse_record("major", "0"),
                sparse_record("minor", "0"),
                sparse_record("realsize", "200"),
            ]
            .concat(),
            "0.0",
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &records)
            .ustar("sparse", b'0', &[1; 20], "", 0o644);
        let bytes = archive.finish();
        let result = decode_first(&bytes, sparse_policy()).await;
        assert!(
            matches!(
                &result,
                Err(DecodeError::Framing(FrameError {
                    inner: FrameErrorInner::UnsupportedSparseFormat { version },
                    ..
                })) if version == expected
            ),
            "{expected}: {:?}",
            result.err()
        );
    }
    Ok(())
}

#[tokio::test]
async fn rejects_malformed_pax_1_0_maps() -> TestResult {
    let records = SparseFile::new(200, &[]).pax_1_0_records("sparse");
    let padded = |map: &[u8], data: &[u8]| {
        let mut payload = map.to_vec();
        payload.resize(BLOCK_SIZE, 0);
        payload.extend_from_slice(data);
        payload
    };
    let mut garbage_padding = padded(b"1\n0\n10\n", &[1; 10]);
    garbage_padding[BLOCK_SIZE - 1] = b'x';
    for (case, payload, expected) in [
        ("garbage padding", garbage_padding, "map padding is not NUL"),
        (
            "leading zero",
            padded(b"1\n00\n0\n", b""),
            "invalid decimal value",
        ),
        (
            "non-decimal",
            padded(b"1\n0x0\n0\n", b""),
            "invalid decimal value",
        ),
        (
            "unterminated line",
            padded(b"1\n0\n0", b""),
            "invalid decimal value",
        ),
        (
            "map beyond payload",
            b"2\n0\n10\n".to_vec(),
            "map extends beyond the member payload",
        ),
        (
            "stored size mismatch",
            padded(b"1\n0\n10\n", &[1; 11]),
            "map entry lengths do not match the stored data size",
        ),
        (
            "empty payload",
            Vec::new(),
            "member payload does not contain its map",
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        archive
            .pax(b'x', &records)
            .ustar("sparse", b'0', &payload, "", 0o644);
        let bytes = archive.finish();
        let result = decode_first(&bytes, sparse_policy()).await;
        assert_eq!(sparse_error(result), Some(expected), "{case}");
    }
    Ok(())
}

#[tokio::test]
async fn rejects_truncated_sparse_maps() -> TestResult {
    // Keep each member header and its first map block.
    for (format, len) in [
        (SparseFormat::OldGnu, 2 * BLOCK_SIZE),
        (SparseFormat::Pax1_0, 4 * BLOCK_SIZE),
    ] {
        let mut archive = ArchiveBuilder::new();
        SparseFile::fragmented().append(&mut archive, format, "sparse");
        let mut bytes = archive.into_unterminated();
        bytes.truncate(len);
        let result = decode_first(&bytes, sparse_policy()).await;
        let Err(DecodeError::Framing(FrameError { inner, .. })) = result else {
            return Err(io::Error::other("expected a framing error").into());
        };
        match format {
            SparseFormat::OldGnu => assert!(matches!(
                inner,
                FrameErrorInner::UnexpectedEof {
                    expected: "old GNU sparse extension block"
                }
            )),
            _ => assert!(matches!(
                inner,
                FrameErrorInner::TruncatedPayload {
                    owner: DataOwner::SparseMap,
                    ..
                }
            )),
        }
    }
    Ok(())
}

#[tokio::test]
async fn bounds_sparse_map_blocks() -> TestResult {
    // The fragmented map needs three old GNU extension blocks or two pax 1.0 map blocks.
    for (format, header_position, map_size) in [
        (SparseFormat::OldGnu, 0, 3 * BLOCK_SIZE as u64),
        (
            SparseFormat::Pax1_0,
            2 * BLOCK_SIZE as u64,
            2 * BLOCK_SIZE as u64,
        ),
    ] {
        let mut archive = ArchiveBuilder::new();
        SparseFile::fragmented().append(&mut archive, format, "sparse");
        let bytes = archive.finish();

        let mut members = TarArchive::new(bytes.as_slice())
            .with_policy(sparse_policy().max_gnu_sparse_map_size(map_size))
            .members();
        assert!(matches!(members.next().await?, Some(Member::File { .. })));

        let result = decode_first(
            &bytes,
            sparse_policy().max_gnu_sparse_map_size(map_size - 1),
        )
        .await;
        assert!(
            matches!(
                result,
                Err(DecodeError::Framing(FrameError {
                    position,
                    inner: FrameErrorInner::SparseMapTooLarge { limit },
                })) if position == header_position && limit == map_size - 1
            ),
            "{format:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn rejects_sparse_records_in_global_pax_headers() -> TestResult {
    let mut archive = ArchiveBuilder::new();
    archive
        .pax(b'g', &sparse_record("name", "global"))
        .ustar("file", b'0', b"", "", 0o644);
    let bytes = archive.finish();

    let result = decode_first(&bytes, sparse_policy()).await;
    assert_eq!(
        sparse_error(result),
        Some("GNU sparse records cannot appear in a global pax header")
    );
    Ok(())
}
