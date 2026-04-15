//! Batch existence proof verification with shared intermediate hash caching.
//!
//! When multiple existence proofs verify against the same root, proofs for keys
//! with shared prefixes traverse identical internal nodes. This module caches
//! those intermediate hashes so each unique internal node is computed at most once.

use alloc::vec::Vec;
use anyhow::{ensure, Result};
use borsh::{BorshDeserialize, BorshSerialize};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{
    proof::{SparseMerkleInternalNode, SparseMerkleProof},
    Bytes32Ext, KeyHash, RootHash, SimpleHasher, ValueHash,
};

/// A batch of existence proofs to verify against the same root.
/// Entries are individual `SparseMerkleProof`s with their key and value.
#[derive(Debug, Clone, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct BatchExistenceProof<H: SimpleHasher> {
    // Borsh derive otherwise adds a spurious `H: BorshSerialize` bound; the
    // actual serialization depends only on `SparseMerkleProof<H>` (whose own
    // borsh impl is H-free thanks to the phantom bound override).
    #[borsh(bound(serialize = "", deserialize = ""))]
    pub entries: Vec<BatchExistenceEntry<H>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct BatchExistenceEntry<H: SimpleHasher> {
    pub key_hash: KeyHash,
    pub value: Vec<u8>,
    #[borsh(bound(serialize = "", deserialize = ""))]
    pub proof: SparseMerkleProof<H>,
}

impl<H: SimpleHasher> BatchExistenceProof<H> {
    pub fn new(entries: Vec<BatchExistenceEntry<H>>) -> Self {
        Self { entries }
    }

    /// Verify all entries exist in the tree with the given root.
    /// Uses intermediate hash caching to avoid redundant computations
    /// at shared internal nodes.
    ///
    /// Returns Ok(()) if all entries are verified, or an error describing
    /// which entry failed.
    pub fn verify(&self, expected_root: RootHash) -> Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }

        // Cache: bit_prefix -> computed hash at that node.
        // The bit prefix is the path from root to the node, encoded as
        // (prefix_bit_length: u16 LE, prefix_bytes: ceil(bits/8)).
        // This uniquely identifies every node in the tree.
        let mut cache: HashMap<Vec<u8>, [u8; 32]> = HashMap::new();

        // Seed cache with expected root at the 0-bit prefix, so every proof's
        // final ascent terminates in a cache hit against the root.
        let root_key = encode_cache_key(0, &[0u8; 32]);
        cache.insert(root_key, expected_root.0);

        for (entry_idx, entry) in self.entries.iter().enumerate() {
            let leaf = entry.proof.leaf().ok_or_else(|| {
                anyhow::anyhow!("Entry {}: missing leaf in existence proof", entry_idx)
            })?;

            ensure!(
                entry.key_hash == leaf.key_hash,
                "Entry {}: key mismatch. Proof key: {:?}, expected: {:?}",
                entry_idx,
                leaf.key_hash,
                entry.key_hash
            );

            let expected_value_hash = ValueHash::with::<H>(&entry.value);
            ensure!(
                expected_value_hash == leaf.value_hash,
                "Entry {}: value hash mismatch",
                entry_idx
            );

            let depth = entry.proof.siblings().len();
            ensure!(
                depth <= 256,
                "Entry {}: proof depth {} exceeds 256",
                entry_idx,
                depth
            );

            let mut current_hash = leaf.hash::<H>();

            // Match the bit ordering used by SparseMerkleProof::verify():
            // iter_bits() is MSB-first, .rev() makes it LSB-first,
            // .skip(256 - depth) leaves bits at positions (depth-1)..=0
            // in reversed (deepest-first) order.
            let bits: Vec<bool> = entry
                .key_hash
                .0
                .iter_bits()
                .rev()
                .skip(256 - depth)
                .collect();

            let mut early_exit = false;

            for (i, sibling) in entry.proof.siblings().iter().enumerate() {
                let bit = bits[i];
                let sibling_hash = sibling.hash::<H>();

                current_hash = if bit {
                    SparseMerkleInternalNode::new(sibling_hash, current_hash).hash::<H>()
                } else {
                    SparseMerkleInternalNode::new(current_hash, sibling_hash).hash::<H>()
                };

                // After step i, current_hash is the hash of the subtree at level
                // (depth - 1 - i) from the root, identified by the first
                // (depth - 1 - i) bits of the key hash (MSB-first).
                let level_from_root = depth - 1 - i;
                let cache_key = encode_cache_key(level_from_root, &entry.key_hash.0);

                if let Some(&cached_hash) = cache.get(&cache_key) {
                    ensure!(
                        current_hash == cached_hash,
                        "Entry {}: hash mismatch at level {} from root. \
                         Computed {:?}, cached {:?}.",
                        entry_idx,
                        level_from_root,
                        current_hash,
                        cached_hash
                    );
                    early_exit = true;
                    break;
                } else {
                    cache.insert(cache_key, current_hash);
                }
            }

            if !early_exit {
                ensure!(
                    current_hash == expected_root.0,
                    "Entry {}: root hash mismatch. Computed {:?}, expected {:?}",
                    entry_idx,
                    current_hash,
                    expected_root.0
                );
            }
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Encode a cache key from a bit prefix length and key hash bytes.
/// Format: [prefix_bit_length as u16 LE] ++ [first ceil(bits/8) bytes of key, masked]
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

/// Build a `BatchExistenceProof` from individual proofs.
pub fn build_batch_existence_proof<H: SimpleHasher>(
    entries: Vec<(KeyHash, Vec<u8>, SparseMerkleProof<H>)>,
) -> BatchExistenceProof<H> {
    let entries = entries
        .into_iter()
        .map(|(key_hash, value, proof)| BatchExistenceEntry {
            key_hash,
            value,
            proof,
        })
        .collect();
    BatchExistenceProof { entries }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockTreeStore;
    use crate::JellyfishMerkleTree;
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::vec;
    use sha2::Sha256;

    #[test]
    fn test_batch_existence_basic() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let keys: Vec<KeyHash> = (0..10)
            .map(|i| KeyHash::with::<Sha256>(format!("key_{}", i).as_bytes()))
            .collect();
        let values: Vec<Vec<u8>> = (0..10)
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
            let (val, proof) = tree.get_with_proof(*k, 0).expect("get_with_proof");
            assert!(val.is_some());
            proof
                .verify_existence(root, *k, v)
                .expect("individual verify");
            entries.push((*k, v.clone(), proof));
        }

        let batch_proof = build_batch_existence_proof::<Sha256>(entries);
        batch_proof.verify(root).expect("batch verify should pass");
    }

    #[test]
    fn test_batch_existence_large() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let n = 100;
        let keys: Vec<KeyHash> = (0..n)
            .map(|i| KeyHash::with::<Sha256>(format!("user_{:04}", i).as_bytes()))
            .collect();
        let values: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("balance_{}", i * 100).into_bytes())
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
        batch_proof.verify(root).expect("batch verify 100 keys");
    }

    #[test]
    fn test_batch_existence_subset() {
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
        for i in 0..50 {
            let (_, proof) = tree.get_with_proof(keys[i], 0).expect("get_with_proof");
            entries.push((keys[i], values[i].clone(), proof));
        }

        let batch_proof = build_batch_existence_proof::<Sha256>(entries);
        batch_proof.verify(root).expect("batch verify subset");
    }

    #[test]
    fn test_batch_existence_single() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"only_key");
        let value = b"only_value".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");

        let batch_proof = build_batch_existence_proof::<Sha256>(vec![(key, value, proof)]);
        batch_proof.verify(root).expect("single entry batch");
    }

    #[test]
    fn test_batch_existence_wrong_value_fails() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"test_key");
        let value = b"correct_value".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");

        let wrong_value = b"wrong_value".to_vec();
        let batch_proof =
            build_batch_existence_proof::<Sha256>(vec![(key, wrong_value, proof)]);
        assert!(batch_proof.verify(root).is_err());
    }

    #[test]
    fn test_batch_existence_wrong_root_fails() {
        let store = MockTreeStore::default();
        let tree = JellyfishMerkleTree::<_, Sha256>::new(&store);

        let key = KeyHash::with::<Sha256>(b"test_key");
        let value = b"value".to_vec();

        let mut v0 = BTreeMap::new();
        v0.insert(key, Some(value.clone()));

        let (root, batch) = tree.put_value_set(v0, 0).expect("insert");
        store.write_tree_update_batch(batch).expect("write");

        let (_, proof) = tree.get_with_proof(key, 0).expect("get_with_proof");

        let batch_proof = build_batch_existence_proof::<Sha256>(vec![(key, value, proof)]);
        let wrong_root = RootHash([99u8; 32]);
        assert!(batch_proof.verify(wrong_root).is_err());
    }

    #[test]
    fn test_batch_existence_empty() {
        let batch_proof = BatchExistenceProof::<Sha256> { entries: vec![] };
        let root = RootHash([0u8; 32]);
        batch_proof.verify(root).expect("empty batch should pass");
    }

    #[test]
    fn test_encode_cache_key() {
        let key = [
            0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let ck = encode_cache_key(0, &key);
        assert_eq!(ck, vec![0, 0]);

        let ck = encode_cache_key(8, &key);
        assert_eq!(ck, vec![8, 0, 0xAB]);

        let ck = encode_cache_key(4, &key);
        assert_eq!(ck, vec![4, 0, 0xA0]);

        let ck = encode_cache_key(12, &key);
        assert_eq!(ck, vec![12, 0, 0xAB, 0xC0]);

        let key2 = [
            0xAB, 0xCD, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let ck1 = encode_cache_key(12, &key);
        let ck2 = encode_cache_key(12, &key2);
        assert_eq!(ck1, ck2);
    }
}
