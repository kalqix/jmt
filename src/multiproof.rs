//! JMT-style delta multiproof for verifying a transition from R0 to Rf where only specified keys change.
//! Uses SparseMerkleProof for simplicity and robustness, compatible with RocksDB and SP1.

use alloc::vec::Vec;
use core::result::Result as CoreResult;
use std::collections::BTreeMap;
use sha2::Sha256;
use crate::{KeyHash, RootHash, ValueHash, proof::{SparseMerkleNode, SparseMerkleInternalNode, SparseMerkleLeafNode}, SPARSE_MERKLE_PLACEHOLDER_HASH, SimpleHasher};
use crate::proof::SparseMerkleProof;

/// Path encoding: [len_bits: u16 (LE)] + prefix bytes (ceil(len_bits/8)), MSB-first within each byte.
pub type Path = Vec<u8>;

#[derive(Debug)]
pub enum DeltaVerifyError {
    EmptyProof,
    BatchTooLarge,
    InvalidProof { key: [u8; 32], error: &'static str },
    SeedCollision { path_bits: usize },
    MultiRootOld(usize),
    MultiRootNew(usize),
    RootMismatch { got_old: [u8;32], want_old: [u8;32], got_new: [u8;32], want_new: [u8;32] },
}

fn singleton(map: &BTreeMap<Path, [u8; 32]>) -> Option<[u8; 32]> {
    let mut it = map.values();
    let h = *it.next()?;
    if it.next().is_some() { return None; }
    Some(h)
}

/// Check if `prefix` is a prefix of `longer`.
fn path_is_prefix(prefix: &Path, longer: &Path) -> bool {
    let lp = path_len_bits(prefix);
    let ll = path_len_bits(longer);
    if lp > ll { return false; }
    let nbytes = (lp + 7) / 8;
    if nbytes == 0 { return true; }
    let mut ok = true;
    for i in 0..nbytes {
        let pb = prefix[2+i];
        let lb = longer[2+i];
        let same = if i + 1 < nbytes {
            pb == lb
        } else {
            let keep_high = 0xFFu8 << ((nbytes*8) - lp);
            (pb & keep_high) == (lb & keep_high)
        };
        ok &= same;
    }
    ok
}

#[inline]
fn key_bit(key: &KeyHash, bit_idx: usize) -> u8 {
    let byte = bit_idx / 8;
    let off = bit_idx % 8;
    ((key.0[byte] >> (7 - off)) & 1) as u8
}

#[inline]
pub fn path_len_bits(path: &Path) -> usize {
    if path.len() < 2 { return 0; }
    u16::from_le_bytes([path[0], path[1]]) as usize
}

#[inline]
fn set_path_len_bits(path: &mut Path, len_bits: usize) {
    let le = (len_bits as u16).to_le_bytes();
    if path.len() < 2 {
        path.resize(2, 0);
    }
    path[0] = le[0];
    path[1] = le[1];
}

pub fn encode_prefix_from_key(len_bits: usize, key: &KeyHash) -> Path {
    let bytes_len = (len_bits + 7) / 8;
    let mut out = Vec::with_capacity(2 + bytes_len);
    out.extend_from_slice(&(len_bits as u16).to_le_bytes());
    if bytes_len == 0 {
        return out;
    }
    out.extend_from_slice(&key.0[..bytes_len]);
    let extra_bits = (bytes_len * 8).saturating_sub(len_bits);
    if extra_bits > 0 {
        let keep_high = 0xFFu8 << extra_bits;
        let last = out.last_mut().unwrap();
        *last &= keep_high;
    }
    out
}

pub fn append_bit(mut parent: Path, bit: u8) -> Path {
    let parent_bits = path_len_bits(&parent);
    let new_bits = parent_bits + 1;
    let new_bytes = (new_bits + 7) / 8;
    set_path_len_bits(&mut parent, new_bits);
    if parent.len() < 2 + new_bytes {
        parent.resize(2 + new_bytes, 0);
    }
    let bit_pos_from_msb = 7 - ((new_bits - 1) % 8);
    let mask = 1u8 << bit_pos_from_msb;
    if bit != 0 {
        parent[2 + new_bytes - 1] |= mask;
    } else {
        parent[2 + new_bytes - 1] &= !mask;
    }
    parent
}

/// Node hashing helpers
#[inline]
pub fn node_hash_bytes<H: SimpleHasher>(n: &SparseMerkleNode) -> [u8; 32] {
    match n {
        SparseMerkleNode::Null => SPARSE_MERKLE_PLACEHOLDER_HASH,
        SparseMerkleNode::Leaf(leaf) => leaf.hash::<H>(),
        SparseMerkleNode::Internal(internal) => internal.hash::<H>(),
    }
}

#[inline]
pub fn hash_leaf_from_parts<H: SimpleHasher>(key: &KeyHash, value_hash: &[u8; 32]) -> [u8; 32] {
    SparseMerkleLeafNode::new(*key, ValueHash(*value_hash)).hash::<H>()
}

#[inline]
fn hash_internal_from_parts<H: SimpleHasher>(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    SparseMerkleInternalNode::new(*left, *right).hash::<H>()
}

/// Merge hashes for a path with multiple keys
fn merge_hashes<H: SimpleHasher>(path_bits: usize, mut hashes: Vec<(KeyHash, [u8; 32])>) -> [u8; 32] {
    if hashes.len() == 1 {
        return hashes[0].1;
    }
    hashes.sort_by_key(|(k, _)| k.0);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (k, h) in hashes {
        if key_bit(&k, path_bits) == 0 {
            left.push((k, h));
        } else {
            right.push((k, h));
        }
    }
    let left_hash = if left.is_empty() {
        SPARSE_MERKLE_PLACEHOLDER_HASH
    } else {
        merge_hashes::<H>(path_bits + 1, left)
    };
    let right_hash = if right.is_empty() {
        SPARSE_MERKLE_PLACEHOLDER_HASH
    } else {
        merge_hashes::<H>(path_bits + 1, right)
    };
    hash_internal_from_parts::<H>(&left_hash, &right_hash)
}

/// Proof payload types
#[derive(Clone, Debug)]
pub struct DeltaMultiProof<H: SimpleHasher> {
    pub(crate) leaves: Vec<DeltaLeaf<H>>,
}

#[derive(Clone, Debug)]
pub struct DeltaLeaf<H: SimpleHasher> {
    pub key: KeyHash,
    pub old_value: Option<Vec<u8>>, // Raw value for old state verification
    pub new_value_hash: Option<[u8; 32]>, // Hash of new value
    pub proof: SparseMerkleProof<H>, // Proof for old state
}

/// Build delta multiproof from single proofs
pub fn build_delta_multiproof<H: SimpleHasher>(
    proofs: Vec<(KeyHash, Option<Vec<u8>>, Option<Vec<u8>>, SparseMerkleProof<H>)>,
) -> DeltaMultiProof<H> {
    let leaves = proofs.into_iter().map(|(key, old_value, new_value, proof)| {
        DeltaLeaf {
            key,
            old_value,
            new_value_hash: new_value.map(|v| value_hash::<H>(&v)),
            proof,
        }
    }).collect();
    DeltaMultiProof { leaves }
}

/// Compute value hash
pub fn value_hash<H: SimpleHasher>(v: &[u8]) -> [u8; 32] {
    ValueHash::with::<H>(v).0
}

/// Verifier: Recompute R0 and Rf, ensuring only specified keys changed
pub fn verify_delta_multiproof_debug<H: SimpleHasher>(
    _hasher: &H,
    r0: RootHash,
    rf: RootHash,
    proof: &DeltaMultiProof<H>,
) -> Result<(), DeltaVerifyError> {
    if proof.leaves.is_empty() { return Err(DeltaVerifyError::EmptyProof); }
    if proof.leaves.len() > 1024 { return Err(DeltaVerifyError::BatchTooLarge); }

    // Verify old proofs and compute new leaf hashes
    let mut new_leaf_by_key: BTreeMap<KeyHash, [u8; 32]> = BTreeMap::new();
    for dl in &proof.leaves {
        // Verify old proof with raw value
        dl.proof.verify(r0, dl.key, dl.old_value.clone()).map_err(|_| {
            DeltaVerifyError::InvalidProof { key: dl.key.0, error: "Old proof verification failed" }
        })?;
        // Compute new leaf hash
        if let Some(vh_new) = dl.new_value_hash {
            let new_hash = hash_leaf_from_parts::<H>(&dl.key, &vh_new);
            new_leaf_by_key.insert(dl.key, new_hash);
            println!("Key: {:02x?}, New leaf hash: {:02x?}", dl.key.0, new_hash);
        }
    }

    // Collect hashes for each path
    let mut old_path_hashes: BTreeMap<Path, Vec<(KeyHash, [u8; 32])>> = BTreeMap::new();
    let mut new_path_hashes: BTreeMap<Path, Vec<(KeyHash, [u8; 32])>> = BTreeMap::new();

    for dl in &proof.leaves {
        let l_bits = dl.proof.siblings().len() * 4;
        let leaf_path = encode_prefix_from_key(l_bits, &dl.key);
        let old_hash = dl.old_value.as_ref()
            .map(|v| hash_leaf_from_parts::<H>(&dl.key, &value_hash::<H>(v)))
            .unwrap_or(SPARSE_MERKLE_PLACEHOLDER_HASH);
        let new_hash = new_leaf_by_key.get(&dl.key).copied()
            .unwrap_or(SPARSE_MERKLE_PLACEHOLDER_HASH);

        let mut current_old_hash = old_hash;
        let mut current_new_hash = new_hash;

        // Traverse siblings to build path
        for (i, sibling) in dl.proof.siblings().iter().enumerate() {
            let parent_bits = l_bits - (i + 1) * 4;
            let parent_path = encode_prefix_from_key(parent_bits, &dl.key);
            let is_left = key_bit(&dl.key, parent_bits) == 0;

            let sibling_hash = node_hash_bytes::<H>(sibling);
            let (left_hash, right_hash) = if is_left {
                (current_old_hash, sibling_hash)
            } else {
                (sibling_hash, current_old_hash)
            };
            let (new_left_hash, new_right_hash) = if is_left {
                (current_new_hash, sibling_hash)
            } else {
                (sibling_hash, current_new_hash)
            };

            current_old_hash = hash_internal_from_parts::<H>(&left_hash, &right_hash);
            current_new_hash = hash_internal_from_parts::<H>(&new_left_hash, &new_right_hash);

            println!("Key: {:02x?}, Path: {:02x?}, Old: {:02x?}, New: {:02x?}", dl.key.0, parent_path, current_old_hash, current_new_hash);

            old_path_hashes.entry(parent_path.clone()).or_default().push((dl.key, current_old_hash));
            new_path_hashes.entry(parent_path).or_default().push((dl.key, current_new_hash));
        }
    }

    // Merge hashes for each path
    let mut old_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();
    let mut new_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();

    for (path, hashes) in old_path_hashes {
        let path_bits = path_len_bits(&path);
        let merged_hash = merge_hashes::<H>(path_bits, hashes);
        if old_at.insert(path.clone(), merged_hash).is_some() {
            return Err(DeltaVerifyError::SeedCollision { path_bits });
        }
    }

    for (path, hashes) in new_path_hashes {
        let path_bits = path_len_bits(&path);
        let merged_hash = merge_hashes::<H>(path_bits, hashes);
        if new_at.insert(path, merged_hash).is_some() {
            return Err(DeltaVerifyError::SeedCollision { path_bits });
        }
    }

    // Check roots
    let h0 = singleton(&old_at).ok_or_else(|| DeltaVerifyError::MultiRootOld(old_at.len()))?;
    let hf = singleton(&new_at).ok_or_else(|| DeltaVerifyError::MultiRootNew(new_at.len()))?;
    if h0 != r0.0 || hf != rf.0 {
        return Err(DeltaVerifyError::RootMismatch { got_old: h0, want_old: r0.0, got_new: hf, want_new: rf.0 });
    }
    Ok(())
}

pub fn verify_delta_multiproof<H: SimpleHasher>(
    hasher: &H,
    r0: RootHash,
    rf: RootHash,
    proof: &DeltaMultiProof<H>,
) -> bool {
    verify_delta_multiproof_debug::<H>(hasher, r0, rf, proof).is_ok()
}