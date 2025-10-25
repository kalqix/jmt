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

#[derive(Debug)]
pub enum DeltaVerifyError {
    EmptyProof,
    FrameNormalization,
    // We tried to seed the same anchor twice (shouldn't happen with grouping).
    SeedCollision { path_bits: usize },
    // A frame expected a child that was neither seeded (in-union) nor provided as an outside hash.
    MissingChildOld { parent_bits: usize, which: &'static str },
    MissingChildNew { parent_bits: usize, which: &'static str },
    // After bubbling we didn't reduce to a single hash.
    MultiRootOld(usize),
    MultiRootNew(usize),
    // Final roots didn't match the expected ones.
    RootMismatch { got_old: [u8;32], want_old: [u8;32], got_new: [u8;32], want_new: [u8;32] },
}

fn path_is_prefix(prefix: &Path, longer: &Path) -> bool {
    let lp = path_len_bits(prefix);
    let ll = path_len_bits(longer);
    if lp > ll { return false; }
    // Compare the bytes that carry the prefix bits
    let nbytes = (lp + 7) / 8;
    if nbytes == 0 { return true; }
    let mut ok = true;
    for i in 0..nbytes {
        let pb = prefix[2+i];
        let lb = longer[2+i];
        let same = if i + 1 < nbytes {
            pb == lb
        } else {
            // last (partial) byte: mask off trailing bits
            let keep_high = 0xFFu8 << ((nbytes*8) - lp);
            (pb & keep_high) == (lb & keep_high)
        };
        ok &= same;
    }
    ok
}

fn side_has_union(node_path: &Path, left_path: &Path, right_path: &Path, leaf_paths: &[Path]) -> (bool, bool) {
    let mut has_left = false;
    let mut has_right = false;
    for lp in leaf_paths {
        if path_is_prefix(left_path,  lp) { has_left  = true; }
        if path_is_prefix(right_path, lp) { has_right = true; }
        if has_left && has_right { break; }
    }
    (has_left, has_right)
}

fn scrub_outside_hashes_with_leaf_paths(
    mut frames: Vec<InternalFrame>,
    leaf_paths: &[Path],
) -> Vec<InternalFrame> {
    for fr in frames.iter_mut() {
        // if any leaf_path is under left_path, that side is "in union"
        let mut left_has_union  = false;
        let mut right_has_union = false;

        for lp in leaf_paths {
            if path_is_prefix(&fr.left_path,  lp) { left_has_union  = true; }
            if path_is_prefix(&fr.right_path, lp) { right_has_union = true; }
            if left_has_union && right_has_union { break; }
        }

        if left_has_union  { fr.left_outside_hash  = None; }
        if right_has_union { fr.right_outside_hash = None; }
    }
    frames
}

#[inline]
fn key_bit(key: &KeyHash, bit_idx: usize) -> u8 {
    let byte = bit_idx / 8;
    let off  = bit_idx % 8;
    ((key.0[byte] >> (7 - off)) & 1) as u8
}

fn bits(path: &Path) -> usize { path_len_bits(path) }

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
    pub old_base_at_leaf_path: Option<[u8; 32]>,
    pub conflicting_key: Option<KeyHash>,        //key of conflicting leaf (if any)
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

/// Build the compressed-subtree root for a set of leaves under an anchor prefix.
// If a subset has exactly 1 leaf, return that leaf hash (compressed SMT rule).
fn subtree_root_for_keys<H: SimpleHasher>(
    anchor_len: usize,
    mut leaves: Vec<(KeyHash, [u8; 32])>,
) -> [u8; 32] {
    // dedup keys if present twice (e.g., conflicting key added separately)
    use std::collections::BTreeMap;
    let mut uniq: BTreeMap<KeyHash, [u8;32]> = BTreeMap::new();
    for (k,h) in leaves.drain(..) { uniq.insert(k, h); }

    fn build<H: SimpleHasher>(d: usize, items: &[(KeyHash, [u8;32])]) -> [u8;32] {
        if items.len() == 1 {
            return items[0].1;
        }
        let mut left: Vec<(KeyHash, [u8;32])> = Vec::new();
        let mut right: Vec<(KeyHash, [u8;32])> = Vec::new();
        for (k,h) in items {
            if key_bit(k, d) == 0 { left.push((*k,*h)); } else { right.push((*k,*h)); }
        }
        let lh = if left.is_empty()  { SPARSE_MERKLE_PLACEHOLDER_HASH } else { build::<H>(d+1, &left) };
        let rh = if right.is_empty() { SPARSE_MERKLE_PLACEHOLDER_HASH } else { build::<H>(d+1, &right) };
        hash_internal_from_parts::<H>(&lh, &rh)
    }

    let pairs: Vec<(KeyHash, [u8;32])> = uniq.into_iter().collect();
    if pairs.is_empty() { return SPARSE_MERKLE_PLACEHOLDER_HASH; }
    if pairs.len() == 1 { return pairs[0].1; }
    build::<H>(anchor_len, &pairs)
}

pub fn verify_delta_multiproof_debug<H: SimpleHasher>(
    _hasher: &H,
    r0: RootHash,
    rf: RootHash,
    proof: &DeltaMultiProof,
) -> Result<(), DeltaVerifyError> {
    if proof.leaves.is_empty() {
        return Err(DeltaVerifyError::EmptyProof);
    }

    // 1) Normalize frames (merge duplicates; zero out outside-hash sides that are in-union).
    let mut frames = normalize_frames(&proof.internals).map_err(|_| DeltaVerifyError::FrameNormalization)?;
    frames.sort_unstable_by(|a, b| {
        let la = bits(&a.node_path);
        let lb = bits(&b.node_path);
        lb.cmp(&la).then_with(|| a.node_path.cmp(&b.node_path))
    });

    // NEW: collect leaf_paths and scrub
    let leaf_paths: Vec<Path> = proof.leaves.iter().map(|dl| dl.leaf_path.clone()).collect();
    frames = scrub_outside_hashes_with_leaf_paths(frames, &leaf_paths);

    // Precompute NEW leaf hashes for keys that have new_value_hash (used when the conflicting key also updates).
    use std::collections::BTreeMap;
    let mut new_leaf_by_key: BTreeMap<KeyHash, [u8; 32]> = BTreeMap::new();
    for dl in &proof.leaves {
        if let Some(vh_new) = dl.new_value_hash {
            new_leaf_by_key.insert(dl.key, hash_leaf_from_parts::<H>(&dl.key, &vh_new));
        }
    }

    // Group by anchor (leaf_path).
    let mut groups: BTreeMap<Path, Vec<&DeltaLeaf>> = BTreeMap::new();
    for dl in &proof.leaves {
        groups.entry(dl.leaf_path.clone()).or_default().push(dl);
    }

    // 2) Seed anchors: compute OLD/NEW subtree roots under each anchor exactly once.
    let mut old_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();
    let mut new_at: BTreeMap<Path, [u8; 32]> = BTreeMap::new();

    for (leaf_path, dls) in groups {
        let anchor_len = bits(&leaf_path);

        // OLD
        let mut old_pairs: Vec<(KeyHash, [u8;32])> = Vec::new();
        let mut conf_key: Option<KeyHash> = None;
        let mut conf_old: Option<[u8;32]> = None;

        for dl in &dls {
            if let Some(vh_old) = dl.old_value_hash {
                old_pairs.push((dl.key, hash_leaf_from_parts::<H>(&dl.key, &vh_old)));
            } else if let (Some(base), Some(kc)) = (dl.old_base_at_leaf_path, dl.conflicting_key) {
                // only add the conflicting leaf once per anchor
                if conf_key.is_none() {
                    conf_key = Some(kc);
                    conf_old = Some(base);
                    old_pairs.push((kc, base));
                }
            }
            // empty-subtree variant contributes nothing to OLD
        }

        let h_old = if old_pairs.is_empty() {
            SPARSE_MERKLE_PLACEHOLDER_HASH
        } else if old_pairs.len() == 1 {
            old_pairs[0].1
        } else {
            subtree_root_for_keys::<H>(anchor_len, old_pairs)
        };

        // NEW
        let mut new_pairs: Vec<(KeyHash, [u8;32])> = Vec::new();

        // If a conflicting leaf existed at OLD, include it at NEW with updated hash if it changed.
        if let (Some(kc), Some(hc_old)) = (conf_key, conf_old) {
            let hc_new = new_leaf_by_key.get(&kc).copied().unwrap_or(hc_old);
            new_pairs.push((kc, hc_new));
        }
        for dl in &dls {
            if let Some(vh_new) = dl.new_value_hash {
                new_pairs.push((dl.key, hash_leaf_from_parts::<H>(&dl.key, &vh_new)));
            }
            // delete contributes nothing
        }

        let h_new = if new_pairs.is_empty() {
            SPARSE_MERKLE_PLACEHOLDER_HASH
        } else if new_pairs.len() == 1 {
            new_pairs[0].1
        } else {
            subtree_root_for_keys::<H>(anchor_len, new_pairs)
        };

        if old_at.insert(leaf_path.clone(), h_old).is_some() {
            return Err(DeltaVerifyError::SeedCollision { path_bits: anchor_len });
        }
        if new_at.insert(leaf_path, h_new).is_some() {
            return Err(DeltaVerifyError::SeedCollision { path_bits: anchor_len });
        }
    }

    // 3) Bubble up along frames using the same boundaries for OLD and NEW.
    for fr in frames {
        let pb = bits(&fr.node_path);

        // OLD
        let lo = if let Some(h) = old_at.remove(&fr.left_path) {
            h
        } else if let Some(h) = fr.left_outside_hash {
            h
        } else {
            return Err(DeltaVerifyError::MissingChildOld { parent_bits: pb, which: "left" });
        };
        let ro = if let Some(h) = old_at.remove(&fr.right_path) {
            h
        } else if let Some(h) = fr.right_outside_hash {
            h
        } else {
            return Err(DeltaVerifyError::MissingChildOld { parent_bits: pb, which: "right" });
        };

        // NEW
        let ln = if let Some(h) = new_at.remove(&fr.left_path) {
            h
        } else if let Some(h) = fr.left_outside_hash {
            h
        } else {
            return Err(DeltaVerifyError::MissingChildNew { parent_bits: pb, which: "left" });
        };
        let rn = if let Some(h) = new_at.remove(&fr.right_path) {
            h
        } else if let Some(h) = fr.right_outside_hash {
            h
        } else {
            return Err(DeltaVerifyError::MissingChildNew { parent_bits: pb, which: "right" });
        };

        let po = hash_internal_from_parts::<H>(&lo, &ro);
        let pn = hash_internal_from_parts::<H>(&ln, &rn);

        old_at.insert(fr.node_path.clone(), po);
        new_at.insert(fr.node_path, pn);
    }

    // 4) Singletons + root check
    fn singleton(map: &BTreeMap<Path, [u8; 32]>) -> Option<[u8; 32]> {
        let mut it = map.values();
        let h = *it.next()?;
        if it.next().is_some() { return None; }
        Some(h)
    }
    let h0 = singleton(&old_at).ok_or_else(|| DeltaVerifyError::MultiRootOld(old_at.len()))?;
    let hf = singleton(&new_at).ok_or_else(|| DeltaVerifyError::MultiRootNew(new_at.len()))?;
    if h0 != r0.0 || hf != rf.0 {
        return Err(DeltaVerifyError::RootMismatch { got_old: h0, want_old: r0.0, got_new: hf, want_new: rf.0 });
    }
    Ok(())
}

// Keep a bool wrapper if you like:
pub fn verify_delta_multiproof<H: SimpleHasher>(
    hasher: &H,
    r0: RootHash,
    rf: RootHash,
    proof: &DeltaMultiProof,
) -> bool {
    verify_delta_multiproof_debug::<H>(hasher, r0, rf, proof).is_ok()
}