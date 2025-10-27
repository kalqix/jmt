//! Storage-backed integration test for the JMT delta multiproof.
//!
//! Builds a JMT in memory, commits an initial state R0, fetches single proofs,
//! creates a DeltaMultiProof, and verifies the transition to Rf.

use std::collections::{BTreeMap, HashMap};
use digest::Mac;
use hashbrown::HashMap as FastMap;
use sha2::Sha256;
use crate::{JellyfishMerkleTree, KeyHash, OwnedValue, SimpleHasher, ValueHash};
use crate::mock::MockTreeStore;
use crate::multiproof::{build_delta_multiproof, encode_prefix_from_key, verify_delta_multiproof_debug, DeltaLeaf, DeltaMultiProof};
use crate::writer::TreeWriter;

/// Compute JMT value hash.
pub fn value_hash(v: &[u8]) -> [u8; 32] {
    ValueHash::with::<Sha256>(v).0
}

/// Map balances to bytes for JMT, sorting by asset ID.
pub fn map_to_bytes_le(map: &HashMap<i32, i32>, _deterministic: bool) -> Option<Vec<u8>> {
    if map.is_empty() {
        return None;
    }
    let mut entries: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|&(k, _)| k);
    let mut out = Vec::with_capacity(entries.len() * 8);
    for (k, v) in entries {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    Some(out)
}

/// Basic test: Update two keys (alice, bob).
#[test]
fn delta_multiproof_integration_pass() {
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";
    let v_alice_new = b"150";
    let v_bob_new = b"400";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.to_vec()));
    v0.insert(k_bob, Some(v_bob_old.to_vec()));
    v0.insert(k_carol, Some(v_carol_old.to_vec()));
    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    let (val_a_old, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (val_b_old, proof_b_r0) = jmt.get_with_proof(k_bob, 0).expect("proof B@R0");

    proof_a_r0.verify(r0, k_alice, val_a_old.clone()).expect("verify A@R0");
    proof_b_r0.verify(r0, k_bob, val_b_old.clone()).expect("verify B@R0");

    let delta = build_delta_multiproof::<Sha256>(vec![
        (k_alice, val_a_old, Some(v_alice_new.to_vec()), proof_a_r0),
        (k_bob, val_b_old, Some(v_bob_new.to_vec()), proof_b_r0),
    ]);

    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    v1.insert(k_carol, Some(v_carol_old.to_vec()));
    let (rf, tree_update_batch) = jmt.put_value_set(v1, 1).expect("commit v1");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v1");

    verify_delta_multiproof_debug::<Sha256>(&Sha256::new(), r0, rf, &delta).expect("delta multiproof should verify");
}

/// Test non-existence proof: Insert a new key (dave) and update existing keys.
#[test]
fn delta_multiproof_integration_none_pass() {
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");
    let k_dave = KeyHash::with::<Sha256>(b"dave");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";
    let v_alice_new = b"150";
    let v_bob_new = b"400";
    let v_dave_new = b"200";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.to_vec()));
    v0.insert(k_bob, Some(v_bob_old.to_vec()));
    v0.insert(k_carol, Some(v_carol_old.to_vec()));
    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    let (val_a_old, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (val_b_old, proof_b_r0) = jmt.get_with_proof(k_bob, 0).expect("proof B@R0");
    let (val_d_old, proof_d_r0) = jmt.get_with_proof(k_dave, 0).expect("proof D@R0");

    proof_a_r0.verify(r0, k_alice, val_a_old.clone()).expect("verify A@R0");
    proof_b_r0.verify(r0, k_bob, val_b_old.clone()).expect("verify B@R0");
    proof_d_r0.verify_nonexistence(r0, k_dave).expect("verify D@R0");

    let delta = build_delta_multiproof::<Sha256>(vec![
        (k_alice, val_a_old, Some(v_alice_new.to_vec()), proof_a_r0),
        (k_bob, val_b_old, Some(v_bob_new.to_vec()), proof_b_r0),
        (k_dave, val_d_old, Some(v_dave_new.to_vec()), proof_d_r0),
    ]);

    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    v1.insert(k_dave, Some(v_dave_new.to_vec()));
    v1.insert(k_carol, Some(v_carol_old.to_vec()));
    let (rf, tree_update_batch) = jmt.put_value_set(v1, 1).expect("commit v1");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v1");

    verify_delta_multiproof_debug::<Sha256>(&Sha256::new(), r0, rf, &delta).expect("delta multiproof should verify");
}

/// Test with custom keys and balances, similar to batch 63.
#[test]
fn delta_multiproof_integration_kalqix() {
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>("0xA7F1c5F437B9C1A3D8FfDbE6a7E0e2F18Ff8e8c5");
    let k_bob = KeyHash::with::<Sha256>("0x9B21e9B4bC7F6c51c7aBEd32fC8A7bF1E8c4E2b3");
    let k_carol = KeyHash::with::<Sha256>("0xC4a4F0EDe18F12B2b34fAd0F6A0E7b5e7a9D0cF3");
    let k_dave = KeyHash::with::<Sha256>("0x7e4F8B8A3C9aB27F63dE45cAf7c9212fE8D1C4E6");

    let btc = 1;

    let mut user_vs_balances = HashMap::new();
    user_vs_balances.insert(k_alice, HashMap::from([(btc, 100)]));
    user_vs_balances.insert(k_bob, HashMap::from([(btc, 200)]));
    user_vs_balances.insert(k_carol, HashMap::from([(btc, 300)]));

    let mut v0 = BTreeMap::new();
    for (key, value) in user_vs_balances.iter() {
        let v = map_to_bytes_le(value, true);
        v0.insert(*key, v);
    }

    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    let mut user_vs_balances = HashMap::new();
    user_vs_balances.insert(k_dave, HashMap::from([(btc, 200)]));
    user_vs_balances.insert(k_alice, HashMap::from([(btc, 400)]));
    user_vs_balances.insert(k_bob, HashMap::from([(btc, 500)]));

    let mut v1 = BTreeMap::new();
    let mut v1_hash = HashMap::new();
    for (key, value) in user_vs_balances.iter() {
        let v = map_to_bytes_le(value, true);
        v1.insert(*key, v.clone());
        v1_hash.insert(*key, v.unwrap());
    }

    let (rf, tree_update_batch) = jmt.put_value_set(v1, 1).expect("commit v1");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v1");

    let mut proofs = Vec::new();
    for (key, value) in v1_hash.iter() {
        let (val_old, proof) = jmt.get_with_proof(*key, 0).expect("proof");
        proof.verify(r0, *key, val_old.clone()).expect("verify proof");
        proofs.push((*key, val_old, Some(value.clone()), proof));
    }

    let delta = build_delta_multiproof::<Sha256>(proofs);

    for leaf in &delta.leaves {
        println!("Key: {:02x?}, OldValue: {:02x?}, NewHash: {:02x?}", leaf.key.0, leaf.old_value, leaf.new_value_hash);
    }

    verify_delta_multiproof_debug::<Sha256>(&Sha256::new(), r0, rf, &delta).expect("delta multiproof should verify");
}

/// Test failure case: Unrelated key change should fail verification.
#[test]
fn delta_multiproof_integration_fails_if_unrelated_key_changed() {
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";
    let v_alice_new = b"150";
    let v_bob_new = b"400";
    let v_carol_new = b"500";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.to_vec()));
    v0.insert(k_bob, Some(v_bob_old.to_vec()));
    v0.insert(k_carol, Some(v_carol_old.to_vec()));
    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    let (val_a_old, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (val_b_old, proof_b_r0) = jmt.get_with_proof(k_bob, 0).expect("proof B@R0");

    proof_a_r0.verify(r0, k_alice, val_a_old.clone()).expect("verify A@R0");
    proof_b_r0.verify(r0, k_bob, val_b_old.clone()).expect("verify B@R0");

    let delta = build_delta_multiproof::<Sha256>(vec![
        (k_alice, val_a_old, Some(v_alice_new.to_vec()), proof_a_r0),
        (k_bob, val_b_old, Some(v_bob_new.to_vec()), proof_b_r0),
    ]);

    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    v1.insert(k_carol, Some(v_carol_new.to_vec()));
    let (rf_bad, tree_update_batch) = jmt.put_value_set(v1, 1).expect("commit v1_bad");
    let _ = store.write_node_batch(&tree_update_batch.node_batch).expect("commit v1_bad");

    verify_delta_multiproof_debug::<Sha256>(&Sha256::new(), r0, rf_bad, &delta).expect_err("delta multiproof should fail when unrelated key changed");
}