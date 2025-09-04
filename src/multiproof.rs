//! JMT-style delta multiproof (single proof that enforces R0 -> Rf where
//! only the listed keys changed).  Works with compressed paths and your
//! `SparseMerkleProof` API (siblings: &[SparseMerkleNode]).
//!
//! This module includes:
//! - Path helpers (explicit bit-prefix Path format)
//! - Types: DeltaLeaf, InternalFrame, DeltaMultiProof
//! - Builder: frames_from_single_proof_r0 (from single proofs @ R0)
//! - Builder: build_delta_from_r0_single_proofs (union per-key frames)
//! - Verifier: verify_delta_multiproof (recompute R0 and Rf with SAME boundaries)

use alloc::vec::Vec;
use core::result::Result as CoreResult;
use std::collections::BTreeMap;

use crate::{KeyHash, RootHash, ValueHash, proof::{SparseMerkleNode, SparseMerkleInternalNode, SparseMerkleLeafNode}, SPARSE_MERKLE_PLACEHOLDER_HASH, SimpleHasher};

/// ---------------------------------------------------------------------------
/// Path helpers
/// Path encoding = [len_bits: u16 (LE)] + prefix bytes (ceil(len_bits/8)),
/// bits are MSB-first within each byte.
/// ---------------------------------------------------------------------------

pub type Path = Vec<u8>;

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

/// Encode the first `len_bits` of the key into a Path.
pub fn encode_prefix_from_key(len_bits: usize, key: &KeyHash) -> Path {
    let bytes_len = (len_bits + 7) / 8;
    let mut out = Vec::with_capacity(2 + bytes_len);
    out.extend_from_slice(&(len_bits as u16).to_le_bytes());
    if bytes_len == 0 {
        return out;
    }
    out.extend_from_slice(&key.0[..bytes_len]);

    // Zero out unused tail bits in the last byte
    let extra_bits = (bytes_len * 8).saturating_sub(len_bits);
    if extra_bits > 0 {
        let keep_high = 0xFFu8 << extra_bits; // keep the high (8 - extra_bits) bits
        let last = out.last_mut().unwrap();
        *last &= keep_high;
    }
    out
}

/// Append one bit (0/1) to an existing parent path.
pub fn append_bit(mut parent: Path, bit: u8) -> Path {
    let parent_bits = path_len_bits(&parent);
    let new_bits = parent_bits + 1;
    let new_bytes = (new_bits + 7) / 8;

    set_path_len_bits(&mut parent, new_bits);

    if parent.len() < 2 + new_bytes {
        parent.resize(2 + new_bytes, 0);
    }

    // set the new bit in the last byte (MSB-first within a byte)
    let bit_pos_from_msb = 7 - ((new_bits - 1) % 8);
    let mask = 1u8 << bit_pos_from_msb;
    if bit != 0 {
        parent[2 + new_bytes - 1] |= mask;
    } else {
        parent[2 + new_bytes - 1] &= !mask;
    }
    parent
}

/// ---------------------------------------------------------------------------
/// Node hashing helper (siblings are SparseMerkleNode)
// ---------------------------------------------------------------------------

#[inline]
pub fn node_hash_bytes<H: SimpleHasher>(n: &SparseMerkleNode) -> [u8; 32] {
    match n {
        SparseMerkleNode::Null => SPARSE_MERKLE_PLACEHOLDER_HASH,
        SparseMerkleNode::Leaf(leaf) => leaf.hash::<H>(),
        SparseMerkleNode::Internal(internal) => internal.hash::<H>(),
    }
}

/// Leaf node hash from (key, value_hash) using crate’s domain separation.
#[inline]
fn hash_leaf_from_parts<H: SimpleHasher>(key: &KeyHash, value_hash: &[u8; 32]) -> [u8; 32] {
    SparseMerkleLeafNode::new(*key, ValueHash(*value_hash)).hash::<H>()
}

/// Internal node hash from (left, right) using crate’s domain separation.
#[inline]
fn hash_internal_from_parts<H: SimpleHasher>(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    SparseMerkleInternalNode::new(*left, *right).hash::<H>()
}

/// ---------------------------------------------------------------------------
/// Proof payload types
/// ---------------------------------------------------------------------------

/// One touched key in the batch, with its exact leaf path depth (len_bits = siblings.len()).
/// The zkVM should recompute the new_value_hash from balances and compare to enforce correctness.
#[derive(Clone)]
pub struct DeltaLeaf {
    pub key: KeyHash,
    pub leaf_path: Path,                 // len_bits = siblings.len() from the (non-)inclusion proof
    pub old_value_hash: Option<[u8; 32]>,// None = key did not exist in R₀ (insert)
    pub new_value_hash: Option<[u8; 32]>,// None = key removed in Rƒ (delete)
}

#[derive(Clone)]
pub struct InternalFrame {
    pub node_path: Path,                   // parent path (len = k)
    pub left_path: Path,                   // node_path + 0
    pub right_path: Path,                  // node_path + 1
    pub left_outside_hash: Option<[u8;32]>,
    pub right_outside_hash: Option<[u8;32]>,
}

#[derive(Clone)]
pub struct DeltaMultiProof {
    pub leaves: Vec<DeltaLeaf>,            // unique keys, any order (normalized in verify)
    pub internals: Vec<InternalFrame>,     // frames; may contain duplicates (merged in verify)
}

/// ---------------------------------------------------------------------------
/// Build frames from a single proof (siblings are leaf→root)
/// ---------------------------------------------------------------------------

/// Convert a single proof under R0 into bottom→up frames, and return the leaf depth (bits).
/// IMPORTANT: siblings are ordered from the leaf’s immediate sibling up to the root sibling.
/// The bit consumed at step `s` is: `bit_idx = L - 1 - s`, where `L = siblings.len()`.
pub fn frames_from_single_proof_r0<H: SimpleHasher>(
    key: KeyHash,
    siblings_bottom_up: &[SparseMerkleNode],
) -> (Vec<InternalFrame>, usize /* leaf_len_bits */) {
    let l = siblings_bottom_up.len(); // depth for this key
    let mut frames = Vec::with_capacity(l);

    for (s, sib_node) in siblings_bottom_up.iter().enumerate() {
        let bit_idx = l - 1 - s;

        // Parent path is the first `bit_idx` bits.
        let parent_path = encode_prefix_from_key(bit_idx, &key);
        let left_path   = append_bit(parent_path.clone(), 0);
        let right_path  = append_bit(parent_path.clone(), 1);

        // Which side did our path take? Look at key’s bit at index `bit_idx`.
        let byte = bit_idx / 8;
        let off  = bit_idx % 8;
        let my_bit = ((key.0[byte] >> (7 - off)) & 1) as u8;

        let sib_hash = node_hash_bytes::<H>(sib_node);
        let (left_outside_hash, right_outside_hash) = if my_bit == 0 {
            (None, Some(sib_hash))      // we were left; sibling is right (outside)
        } else {
            (Some(sib_hash), None)      // we were right; sibling is left (outside)
        };

        frames.push(InternalFrame {
            node_path: parent_path,
            left_path,
            right_path,
            left_outside_hash,
            right_outside_hash,
        });
    }

    (frames, l)
}


/// ---------------------------------------------------------------------------
/// Verifier (zkVM): recompute R0 and Rf using the *same* boundary frames.
/// If anything outside the union changed, recomputing Rf will fail.
/// ---------------------------------------------------------------------------

fn normalize_frames(frames: &[InternalFrame]) -> CoreResult<Vec<InternalFrame>, ()> {
    let mut by_path: BTreeMap<Path, InternalFrame> = BTreeMap::new();

    for fr in frames {
        match by_path.get_mut(&fr.node_path) {
            None => {
                by_path.insert(fr.node_path.clone(), fr.clone());
            }
            Some(acc) => {
                // Child path consistency
                if acc.left_path != fr.left_path || acc.right_path != fr.right_path {
                    return Err(());
                }

                // Merge outside hashes:
                // If either side is in-union for any key, that side must be None.
                // Otherwise, hashes must match.
                acc.left_outside_hash = match (acc.left_outside_hash, fr.left_outside_hash) {
                    (None,    _)        => None,
                    (_,       None)     => None,
                    (Some(a), Some(b)) if a == b => Some(a),
                    _ => return Err(()),
                };
                acc.right_outside_hash = match (acc.right_outside_hash, fr.right_outside_hash) {
                    (None,    _)        => None,
                    (_,       None)     => None,
                    (Some(a), Some(b)) if a == b => Some(a),
                    _ => return Err(()),
                };
            }
        }
    }

    Ok(by_path.into_values().collect())
}

/// Verifies both roots using the same boundary frames.
/// Returns true iff: old → R0 and new → Rf.
pub fn verify_delta_multiproof<H: SimpleHasher>(
    _hasher: &H,
    r0: RootHash,
    rf: RootHash,
    proof: &DeltaMultiProof,
) -> bool {
    if proof.leaves.is_empty() {
        return false;
    }

    // Defensive: normalize/merge frames
    let mut frames = match normalize_frames(&proof.internals) {
        Ok(v) => v,
        Err(_) => return false,
    };

    // Process deepest-first (longer path first), then lexicographic
    frames.sort_unstable_by(|a, b| {
        let la = path_len_bits(&a.node_path);
        let lb = path_len_bits(&b.node_path);
        lb.cmp(&la).then_with(|| a.node_path.cmp(&b.node_path))
    });

    // Place leaves at their exact leaf paths (depth = siblings.len())
    let mut old_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();
    let mut new_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();

    for dl in &proof.leaves {
        // For R₀: if the key did not exist, use the placeholder at this leaf position.
        let h_old = match dl.old_value_hash {
            Some(vh) => hash_leaf_from_parts::<H>(&dl.key, &vh),
            None     => SPARSE_MERKLE_PLACEHOLDER_HASH,
        };
        // For Rƒ: if the key is deleted, use the placeholder; else the new leaf hash.
        let h_new = match dl.new_value_hash {
            Some(vh) => hash_leaf_from_parts::<H>(&dl.key, &vh),
            None     => SPARSE_MERKLE_PLACEHOLDER_HASH,
        };
        if old_at.insert(dl.leaf_path.clone(), h_old).is_some() { return false; }
        if new_at.insert(dl.leaf_path.clone(), h_new).is_some() { return false; }
    }

    // Bubble up parents using frames
    for fr in frames {
        let (lo, ro) = {
            let left  = match old_at.remove(&fr.left_path)  { Some(h) => h, None => match fr.left_outside_hash { Some(h) => h, None => return false } };
            let right = match old_at.remove(&fr.right_path) { Some(h) => h, None => match fr.right_outside_hash { Some(h) => h, None => return false } };
            (left, right)
        };
        let (ln, rn) = {
            let left  = match new_at.remove(&fr.left_path)  { Some(h) => h, None => match fr.left_outside_hash { Some(h) => h, None => return false } };
            let right = match new_at.remove(&fr.right_path) { Some(h) => h, None => match fr.right_outside_hash { Some(h) => h, None => return false } };
            (left, right)
        };

        let po = hash_internal_from_parts::<H>(&lo, &ro);
        let pn = hash_internal_from_parts::<H>(&ln, &rn);

        if old_at.insert(fr.node_path.clone(), po).is_some() { return false; }
        if new_at.insert(fr.node_path, pn).is_some() { return false; }
    }

    // Must reduce to exactly one hash each (the root hash). We compare the hashes, not the path.
    fn singleton(map: &BTreeMap<Path, [u8; 32]>) -> Option<[u8; 32]> {
        let mut it = map.values();
        let h = *it.next()?;
        if it.next().is_some() { return None; }
        Some(h)
    }

    match (singleton(&old_at), singleton(&new_at)) {
        (Some(h0), Some(hf)) => h0 == r0.0 && hf == rf.0,
        _ => false,
    }
}