//! Flat batch existence proof with pre-hashed siblings.
//!
//! Same security as [`BatchExistenceProof`], but siblings are stored as raw
//! `[u8; 32]` digests instead of `SparseMerkleNode` enums. The verifier does one
//! hash call per tree level instead of two — the sibling hash is computed on
//! the host side when the proof is built.
//!
//! Use [`FlatBatchExistenceProof::from_batch`] to convert from a
//! [`BatchExistenceProof`], pre-hashing all siblings on the host side.

use alloc::vec::Vec;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{
    batch_existence::{BatchExistenceEntry, BatchExistenceProof},
    proof::{SparseMerkleProof, INTERNAL_DOMAIN_SEPARATOR, LEAF_DOMAIN_SEPARATOR},
    Bytes32Ext, KeyHash, SimpleHasher, ValueHash,
};

/// A single existence proof with pre-hashed siblings.
///
/// `sibling_hashes` is ordered identically to `SparseMerkleProof::siblings`:
/// index `0` pairs with the bit at key-hash position `depth-1` and is the
/// deepest sibling (combined with the leaf in the verify fold); index
/// `depth-1` pairs with the top (MSB) bit and is combined last, closest to
/// the root.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct FlatExistenceEntry {
    pub key_hash: KeyHash,
    pub value: Vec<u8>,
    pub sibling_hashes: Vec<[u8; 32]>,
}

/// Batch of flat existence proofs verified against one root with shared
/// intermediate hash caching across entries.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct FlatBatchExistenceProof {
    pub entries: Vec<FlatExistenceEntry>,
}

#[inline]
fn leaf_hash<H: SimpleHasher>(key_hash: &[u8; 32], value_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = H::new();
    hasher.update(LEAF_DOMAIN_SEPARATOR);
    hasher.update(key_hash);
    hasher.update(value_hash);
    hasher.finalize()
}

#[inline]
fn internal_hash<H: SimpleHasher>(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = H::new();
    hasher.update(INTERNAL_DOMAIN_SEPARATOR);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize()
}

/// Encode a cache key identifying a node at a given bit depth from the root.
/// Format matches `batch_existence::encode_cache_key`:
/// `[prefix_bit_length as u16 LE] ++ [ceil(bits/8) bytes of key, masked]`.
fn encode_cache_key(prefix_bit_length: usize, key_bytes: &[u8; 32]) -> Vec<u8> {
    let prefix_byte_count = (prefix_bit_length + 7) / 8;
    let mut cache_key = Vec::with_capacity(2 + prefix_byte_count);
    cache_key.extend_from_slice(&(prefix_bit_length as u16).to_le_bytes());

    if prefix_byte_count > 0 {
        cache_key.extend_from_slice(&key_bytes[..prefix_byte_count]);

        let extra_bits = prefix_byte_count * 8 - prefix_bit_length;
        if extra_bits > 0 {
            let mask = 0xFFu8 << extra_bits;
            let last_idx = cache_key.len() - 1;
            cache_key[last_idx] &= mask;
        }
    }

    cache_key
}

impl FlatExistenceEntry {
    /// Flatten a `BatchExistenceEntry` by pre-hashing all siblings.
    pub fn from_entry<H: SimpleHasher>(entry: &BatchExistenceEntry<H>) -> Self {
        Self::from_proof::<H>(entry.key_hash, entry.value.clone(), &entry.proof)
    }

    /// Flatten a raw `(key_hash, value, proof)` triple.
    pub fn from_proof<H: SimpleHasher>(
        key_hash: KeyHash,
        value: Vec<u8>,
        proof: &SparseMerkleProof<H>,
    ) -> Self {
        let sibling_hashes = proof.siblings().iter().map(|s| s.hash::<H>()).collect();
        FlatExistenceEntry {
            key_hash,
            value,
            sibling_hashes,
        }
    }
}

impl FlatBatchExistenceProof {
    pub fn new(entries: Vec<FlatExistenceEntry>) -> Self {
        Self { entries }
    }

    /// Convert from a `BatchExistenceProof` by pre-hashing all siblings.
    /// This should be called on the host / prover side.
    pub fn from_batch<H: SimpleHasher>(batch: &BatchExistenceProof<H>) -> Self {
        let entries = batch
            .entries
            .iter()
            .map(FlatExistenceEntry::from_entry::<H>)
            .collect();
        FlatBatchExistenceProof { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify all entries exist in the tree with the given root.
    /// Uses intermediate hash caching to avoid recomputing shared subtrees.
    ///
    /// Cost: `2 + depth` hash calls per proof (one for `ValueHash`, one for the
    /// leaf, one per tree level), versus `2 + 2*depth` in
    /// [`BatchExistenceProof::verify`] — roughly a 50% reduction at internal
    /// levels because siblings are already hashed.
    pub fn verify<H: SimpleHasher>(&self, expected_root: [u8; 32]) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let mut cache: HashMap<Vec<u8>, [u8; 32]> = HashMap::new();
        // Seed the root so every ascent terminates in a cache hit.
        cache.insert(encode_cache_key(0, &[0u8; 32]), expected_root);

        for entry in &self.entries {
            let depth = entry.sibling_hashes.len();
            if depth > 256 {
                return false;
            }

            let value_hash = ValueHash::with::<H>(&entry.value);
            let mut current = leaf_hash::<H>(&entry.key_hash.0, &value_hash.0);

            // Match the bit ordering used by SparseMerkleProof::verify():
            // iter_bits() is MSB-first; .rev().skip(256-depth) leaves bits at
            // positions (depth-1)..=0 in deepest-first order.
            let bits: Vec<bool> = entry
                .key_hash
                .0
                .iter_bits()
                .rev()
                .skip(256 - depth)
                .collect();

            let mut early_exit = false;

            for (i, sibling) in entry.sibling_hashes.iter().enumerate() {
                current = if bits[i] {
                    internal_hash::<H>(sibling, &current)
                } else {
                    internal_hash::<H>(&current, sibling)
                };

                // After step i, `current` is the hash of the subtree at level
                // (depth - 1 - i) from the root, identified by the first
                // (depth - 1 - i) MSB bits of the key hash.
                let level_from_root = depth - 1 - i;
                let cache_key = encode_cache_key(level_from_root, &entry.key_hash.0);

                if let Some(&cached) = cache.get(&cache_key) {
                    if current != cached {
                        return false;
                    }
                    early_exit = true;
                    break;
                }
                cache.insert(cache_key, current);
            }

            if !early_exit && current != expected_root {
                return false;
            }
        }

        true
    }
}

impl ArchivedFlatBatchExistenceProof {
    /// Verify the zero-copy archived form. Semantics match
    /// [`FlatBatchExistenceProof::verify`].
    pub fn verify<H: SimpleHasher>(&self, expected_root: &[u8; 32]) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let mut cache: HashMap<Vec<u8>, [u8; 32]> = HashMap::new();
        cache.insert(encode_cache_key(0, &[0u8; 32]), *expected_root);

        for entry in self.entries.iter() {
            let key_bytes: [u8; 32] = entry.key_hash.0;
            let depth = entry.sibling_hashes.len();
            if depth > 256 {
                return false;
            }

            let value_hash = ValueHash::with::<H>(entry.value.as_slice());
            let mut current = leaf_hash::<H>(&key_bytes, &value_hash.0);

            let bits: Vec<bool> = key_bytes.iter_bits().rev().skip(256 - depth).collect();

            let mut early_exit = false;

            for (i, sibling) in entry.sibling_hashes.iter().enumerate() {
                current = if bits[i] {
                    internal_hash::<H>(sibling, &current)
                } else {
                    internal_hash::<H>(&current, sibling)
                };

                let level_from_root = depth - 1 - i;
                let cache_key = encode_cache_key(level_from_root, &key_bytes);

                if let Some(&cached) = cache.get(&cache_key) {
                    if current != cached {
                        return false;
                    }
                    early_exit = true;
                    break;
                }
                cache.insert(cache_key, current);
            }

            if !early_exit && current != *expected_root {
                return false;
            }
        }

        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Verify a single flat existence proof against an expected root.
pub fn verify_flat_existence<H: SimpleHasher>(
    key_hash: &KeyHash,
    value: &[u8],
    sibling_hashes: &[[u8; 32]],
    expected_root: &[u8; 32],
) -> bool {
    let depth = sibling_hashes.len();
    if depth > 256 {
        return false;
    }

    let value_hash = ValueHash::with::<H>(value);
    let mut current = leaf_hash::<H>(&key_hash.0, &value_hash.0);

    let bits: Vec<bool> = key_hash.0.iter_bits().rev().skip(256 - depth).collect();

    for (i, sibling) in sibling_hashes.iter().enumerate() {
        current = if bits[i] {
            internal_hash::<H>(sibling, &current)
        } else {
            internal_hash::<H>(&current, sibling)
        };
    }

    current == *expected_root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_existence::build_batch_existence_proof;
    use crate::mock::MockTreeStore;
    use crate::JellyfishMerkleTree;
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::vec;
    use sha2::Sha256;

    #[test]
    fn test_flat_single_proof() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"test_key");
        let value = b"test_value".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");

        let entry = FlatExistenceEntry::from_proof::<Sha256>(key, value.clone(), &proof);
        assert!(verify_flat_existence::<Sha256>(
            &key,
            &value,
            &entry.sibling_hashes,
            &root.0
        ));
    }

    #[test]
    fn test_flat_matches_batch_existence() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let n = 50;
        let keys: Vec<KeyHash> = (0..n)
            .map(|i| KeyHash::with::<Sha256>(format!("key_{:04}", i).as_bytes()))
            .collect();
        let values: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("value_{}", i).into_bytes())
            .collect();

        let mut v0 = BTreeMap::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            v0.insert(*k, Some(v.clone()));
        }

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let mut entries = Vec::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            let (_, proof) = tree.get_with_proof(*k, 0).expect("get_with_proof");
            entries.push((*k, v.clone(), proof));
        }
        let batch_proof = build_batch_existence_proof::<Sha256>(entries);

        // The BatchExistenceProof verify must pass.
        batch_proof.verify(root).expect("batch verify");

        // The flat form derived from the same proofs must also pass.
        let flat_proof = FlatBatchExistenceProof::from_batch::<Sha256>(&batch_proof);
        assert!(flat_proof.verify::<Sha256>(root.0));
        assert_eq!(flat_proof.len(), n);
    }

    #[test]
    fn test_flat_large_tree_subset() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let n = 1000;
        let keys: Vec<KeyHash> = (0..n)
            .map(|i| KeyHash::with::<Sha256>(format!("key_{:06}", i).as_bytes()))
            .collect();
        let values: Vec<Vec<u8>> = (0..n).map(|i| format!("val_{}", i).into_bytes()).collect();

        let mut v0 = BTreeMap::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            v0.insert(*k, Some(v.clone()));
        }

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let mut entries = Vec::new();
        for i in 0..100 {
            let (_, proof) = tree.get_with_proof(keys[i], 0).expect("get_with_proof");
            entries.push((keys[i], values[i].clone(), proof));
        }
        let batch_proof = build_batch_existence_proof::<Sha256>(entries);
        let flat_proof = FlatBatchExistenceProof::from_batch::<Sha256>(&batch_proof);

        assert!(flat_proof.verify::<Sha256>(root.0));
    }

    #[test]
    fn test_flat_wrong_root_fails() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"key");
        let value = b"val".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (_root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");
        let entries = vec![(key, value, proof)];
        let batch_proof = build_batch_existence_proof::<Sha256>(entries);
        let flat_proof = FlatBatchExistenceProof::from_batch::<Sha256>(&batch_proof);

        assert!(!flat_proof.verify::<Sha256>([99u8; 32]));
    }

    #[test]
    fn test_flat_wrong_value_fails() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"key");
        let value = b"correct".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");

        let sibling_hashes: Vec<[u8; 32]> =
            proof.siblings().iter().map(|s| s.hash::<Sha256>()).collect();

        let wrong_value = b"wrong".to_vec();
        assert!(!verify_flat_existence::<Sha256>(
            &key,
            &wrong_value,
            &sibling_hashes,
            &root.0
        ));
    }

    #[test]
    fn test_flat_empty_batch() {
        let flat = FlatBatchExistenceProof { entries: vec![] };
        assert!(flat.verify::<Sha256>([0u8; 32]));
    }

    #[test]
    fn test_flat_rkyv_roundtrip() {
        use rkyv::rancor::Error;

        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let n = 10;
        let keys: Vec<KeyHash> = (0..n)
            .map(|i| KeyHash::with::<Sha256>(format!("k{}", i).as_bytes()))
            .collect();
        let values: Vec<Vec<u8>> = (0..n).map(|i| format!("v{}", i).into_bytes()).collect();

        let mut v0 = BTreeMap::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            v0.insert(*k, Some(v.clone()));
        }

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let mut entries = Vec::new();
        for (k, v) in keys.iter().zip(values.iter()) {
            let (_, proof) = tree.get_with_proof(*k, 0).expect("get_with_proof");
            entries.push((*k, v.clone(), proof));
        }
        let batch_proof = build_batch_existence_proof::<Sha256>(entries);
        let flat_proof = FlatBatchExistenceProof::from_batch::<Sha256>(&batch_proof);

        let bytes = rkyv::to_bytes::<Error>(&flat_proof).expect("rkyv serialize");
        let archived = rkyv::access::<ArchivedFlatBatchExistenceProof, Error>(&bytes[..])
            .expect("rkyv access");
        assert!(archived.verify::<Sha256>(&root.0));
    }

    #[test]
    fn test_flat_rkyv_wrong_root_fails() {
        use rkyv::rancor::Error;

        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"test_key");
        let value = b"value".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (_root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");
        let batch_proof = build_batch_existence_proof::<Sha256>(vec![(key, value, proof)]);
        let flat_proof = FlatBatchExistenceProof::from_batch::<Sha256>(&batch_proof);

        let bytes = rkyv::to_bytes::<Error>(&flat_proof).expect("rkyv serialize");
        let archived = rkyv::access::<ArchivedFlatBatchExistenceProof, Error>(&bytes[..])
            .expect("rkyv access");

        assert!(!archived.verify::<Sha256>(&[99u8; 32]));
    }
}
