//! Storage-backed integration test for the JMT delta multiproof.
//!
//! This test builds a JMT in memory, commits an initial state R0, fetches single
//! proofs under R0, converts those single proofs into `InternalFrame`s, builds a
//! single `DeltaMultiProof`, then commits a new state Rf and verifies that the
//! SAME internal frames reconstruct both R0 (with old values) and Rf (with new
//! values). It also demonstrates that changing an unrelated key causes the
//! verification to fail.

use std::collections::BTreeMap;
use hashbrown::HashMap;
use proptest::bits::BitSetLike;
use sha2::Sha256;
use crate::{JellyfishMerkleTree, KeyHash, SimpleHasher, ValueHash};
use crate::mock::MockTreeStore;
use crate::multiproof::{ encode_prefix_from_key, frames_from_single_proof_r0, verify_delta_multiproof, DeltaLeaf, DeltaMultiProof, InternalFrame, Path};
use crate::writer::TreeWriter;

/// ---- Helpers to map single proofs -> InternalFrames (R0) --------------------

pub fn map_to_bytes_le(map: &HashMap<i32, i32>, deterministic: bool) -> Option<Vec<u8>> {
    if map.is_empty() {
        return None;
    }

    // Pre-allocate: each entry = 8 bytes (4 for key + 4 for value)
    let mut out = Vec::with_capacity(map.len() * 8);

    if deterministic {
        // Sort by key for stable output (costs O(n log n))
        let mut entries: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by_key(|&(k, _)| k);
        for (k, v) in entries {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        // Fastest: whatever iteration order the HashMap has
        for (&k, &v) in map.iter() {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    Some(out)
}/// Convenience: compute JMT value hash the same way your crate does.
fn value_hash(v: &[u8]) -> [u8; 32] {
    // Was: Sha256::hash(v)
    ValueHash::with::<Sha256>(v).0
}

/// Host-side input for unioning multiple single proofs (all w.r.t. R0).
pub struct SingleForDeltaHost {
    pub key: KeyHash,
    pub old_value_hash: Option<[u8; 32]>, // None for insert
    pub new_value_hash: Option<[u8; 32]>, // None for delete
    pub frames_r0: Vec<InternalFrame>,
    pub leaf_path: Path,
    pub old_base_at_leaf_path: Option<[u8; 32]>,
    pub conflicting_key: Option<KeyHash>,
}

/// Union per-key frames into one DeltaMultiProof. No storage reads; duplicates are merged in verify.
pub fn build_delta_from_r0_single_proofs(
    inputs: Vec<SingleForDeltaHost>,
) -> DeltaMultiProof {
    let mut leaves: Vec<DeltaLeaf> = inputs.iter().map(|it| DeltaLeaf {
        key: it.key,
        leaf_path: it.leaf_path.clone(),
        old_value_hash: it.old_value_hash,
        new_value_hash: it.new_value_hash,
        old_base_at_leaf_path: it.old_base_at_leaf_path,
        conflicting_key: it.conflicting_key,
    }).collect();

    // deterministic order
    leaves.sort_unstable_by(|a, b| a.key.0.cmp(&b.key.0));

    // concatenate frames; verifier will normalize
    let mut internals: Vec<InternalFrame> = Vec::new();
    for it in inputs {
        internals.extend_from_slice(&it.frames_r0.as_slice());
    }

    DeltaMultiProof { leaves, internals }
}

/// ---------- The tests --------------------------------------------------------

#[test]
fn delta_multiproof_integration_pass() {
    // 1) Build a JMT in memory and commit R0 with 3 accounts.
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob   = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";

    let v_alice_new = b"150";
    let v_bob_new = b"400";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.clone().to_vec()));
    v0.insert(k_bob,   Some(v_bob_old.clone().to_vec()));
    v0.insert(k_carol, Some(v_carol_old.clone().to_vec()));
    println!("carol {:?}", v_carol_old.clone().to_vec());
    println!("alice old {:?}", v_alice_old.clone().to_vec());
    println!("bob old {:?}", v_bob_old.clone().to_vec());
    println!("alice new {:?}", v_alice_new.clone().to_vec());
    println!("bob new {:?}", v_bob_new.clone().to_vec());
    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    // 2) Fetch single proofs for the *touched* set S = {alice, bob} under R0.
    let (val_a_old, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (val_b_old, proof_b_r0) = jmt.get_with_proof(k_bob,   0).expect("proof B@R0");

    // Sanity: individual single proofs verify under R0.
    proof_a_r0.verify(r0, k_alice, Some(v_alice_old)).expect("verify A@R0");
    proof_b_r0.verify(r0, k_bob,Some(v_bob_old)).expect("verify B@R0");

    // 3) Convert those single proofs into InternalFrames (bottom→up).
    let (frames_a, la_bits) = frames_from_single_proof_r0::<Sha256>(k_alice, proof_a_r0.siblings());
    let (frames_b, lb_bits) = frames_from_single_proof_r0::<Sha256>(k_bob,   proof_b_r0.siblings());

    // 4) Build the DeltaMultiProof for batch S = {alice(new 150), bob(unchanged)}.
    let delta = build_delta_from_r0_single_proofs(vec![
        SingleForDeltaHost {
            key: k_alice,
            old_value_hash: Some(value_hash(v_alice_old)),
            new_value_hash: Some(value_hash(v_alice_new)),
            frames_r0: frames_a,
            leaf_path: encode_prefix_from_key(la_bits, &k_alice),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
        SingleForDeltaHost {
            key: k_bob,
            old_value_hash: Some(value_hash(v_bob_old)),
            new_value_hash: Some(value_hash(v_bob_new)),
            frames_r0: frames_b,
            leaf_path: encode_prefix_from_key(lb_bits, &k_bob),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
    ]);

    // 5) Apply updates to get Rf: ONLY alice changes to 150.
    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    let (rf, _writes1) = jmt.put_value_set(v1, 1).expect("commit v1");
    store.write_node_batch(&_writes1.node_batch).expect("commit v0");
    println!("alice {:?}", v_alice_new.clone().to_vec());
    println!("bob {:?}", v_bob_new.clone().to_vec());

    let a = jmt.get_with_proof(k_carol, 1).expect("proof A@R0");
    println!("carol new {:?}", a.0);
    let a = jmt.get_with_proof(k_alice, 1).expect("proof A@R0");
    println!("alice new {:?}", a.0);
    let a = jmt.get_with_proof(k_bob, 1).expect("proof A@R0");
    println!("bob new {:?}", a.0);

    let a = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    println!("alice old {:?}", a.0);
    let a = jmt.get_with_proof(k_bob, 0).expect("proof A@R0");
    println!("bob old {:?}", a.0);
    // 6) Verify the delta multiproof (should pass).
    assert!(
        verify_delta_multiproof::<Sha256>(&Sha256::new(), r0, rf, &delta),
        "delta multiproof should verify but did not"
    );
}

#[test]
fn delta_multiproof_integration_none_pass() {
    // 1) Build a JMT in memory and commit R0 with 3 accounts.
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob   = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");
    let k_dave = KeyHash::with::<Sha256>(b"dave");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";

    let v_alice_new = b"150";
    let v_bob_new = b"400";
    let v_dave_new = b"200";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.clone().to_vec()));
    v0.insert(k_bob,   Some(v_bob_old.clone().to_vec()));
    v0.insert(k_carol, Some(v_carol_old.clone().to_vec()));

    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    // 2) Fetch single proofs for the *touched* set S = {alice, bob} under R0.
    let (val_a_old, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (val_b_old, proof_b_r0) = jmt.get_with_proof(k_bob,   0).expect("proof B@R0");
    let (val_d_old, proof_d_r0) = jmt.get_with_proof(k_dave,   0).expect("proof D@R0");

    // Sanity: individual single proofs verify under R0.
    proof_a_r0.verify(r0, k_alice, val_a_old).expect("verify A@R0");
    proof_b_r0.verify(r0, k_bob, val_b_old).expect("verify B@R0");
    proof_d_r0.verify_nonexistence(r0, k_dave).expect("verify D@R0");
    let old_base_at_leaf_path_dave = proof_d_r0.leaf().map(|leaf| leaf.hash::<Sha256>());
    let conflicting_key_dave       = proof_d_r0.leaf().map(|leaf| leaf.key_hash());
    
    // 3) Convert those single proofs into InternalFrames (bottom→up).
    let (frames_a, la_bits) = frames_from_single_proof_r0::<Sha256>(k_alice, proof_a_r0.siblings());
    let (frames_b, lb_bits) = frames_from_single_proof_r0::<Sha256>(k_bob,   proof_b_r0.siblings());
    let (frames_d, ld_bits) = frames_from_single_proof_r0::<Sha256>(k_dave,   proof_d_r0.siblings());

    // 4) Build the DeltaMultiProof for batch S = {alice(new 150), bob(unchanged)}.
    let delta = build_delta_from_r0_single_proofs(vec![
        SingleForDeltaHost {
            key: k_alice,
            old_value_hash: Some(value_hash(v_alice_old)),
            new_value_hash: Some(value_hash(v_alice_new)),
            frames_r0: frames_a,
            leaf_path: encode_prefix_from_key(la_bits, &k_alice),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
        SingleForDeltaHost {
            key: k_bob,
            old_value_hash: Some(value_hash(v_bob_old)),
            new_value_hash: Some(value_hash(v_bob_new)),
            frames_r0: frames_b,
            leaf_path: encode_prefix_from_key(lb_bits, &k_bob),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
        SingleForDeltaHost {
            key: k_dave,
            old_value_hash: None,
            new_value_hash: Some(value_hash(v_dave_new)),
            frames_r0: frames_d,
            leaf_path: encode_prefix_from_key(ld_bits, &k_dave),
            old_base_at_leaf_path: old_base_at_leaf_path_dave,
            conflicting_key: conflicting_key_dave,
        },
    ]);

    // 5) Apply updates to get Rf: ONLY alice changes to 150.
    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    v1.insert(k_dave, Some(v_dave_new.to_vec()));
    let (rf, _writes1) = jmt.put_value_set(v1, 1).expect("commit v1");
    store.write_node_batch(&_writes1.node_batch).expect("commit v1");


    let ok = verify_delta_multiproof::<Sha256>(&Sha256::new(), r0, rf, &delta);
    println!("verify_delta_multiproof: {}", ok);
    assert!(ok, "delta multiproof should verify but did not");
}

#[test]
fn delta_multiproof_integration_kalqix() {
    // 1) Build a JMT in memory and commit R0 with 3 accounts.
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);
    let root = jmt.get_root_hash_option(0);

    println!("root at version 0 before: {:?}", root);
    let k_alice = KeyHash::with::<Sha256>("0xA7F1c5F437B9C1A3D8FfDbE6a7E0e2F18Ff8e8c5");
    let k_bob   = KeyHash::with::<Sha256>("0x9B21e9B4bC7F6c51c7aBEd32fC8A7bF1E8c4E2b3");
    let k_carol = KeyHash::with::<Sha256>("0xC4a4F0EDe18F12B2b34fAd0F6A0E7b5e7a9D0cF3");
    let k_dave = KeyHash::with::<Sha256>("0x7e4F8B8A3C9aB27F63dE45cAf7c9212fE8D1C4E6");

    let btc = 1;
    let ethereum = 2;

    let mut user_vs_balances = HashMap::new();

    {
        let alice_balances = user_vs_balances.entry(k_alice).or_insert_with(|| HashMap::new());
        alice_balances.insert(1, 100);
    }
    {
        let bob_balances = user_vs_balances.entry(k_bob).or_insert_with(|| HashMap::new());
        bob_balances.insert(1, 200);
    }
    {
        let carol_balances = user_vs_balances.entry(k_carol).or_insert_with(|| HashMap::new());
        carol_balances.insert(1, 300);
    }

    let mut v0 = BTreeMap::new();

    for(key, value) in user_vs_balances.iter() {
        let v = map_to_bytes_le(value, false);
        v0.insert(key.clone(), v);
    }

    let (r0, tree_update_batch) = jmt.put_value_set(v0.clone(), 0).expect("commit v0");
    store.write_node_batch(&tree_update_batch.node_batch).expect("commit v0");

    let root = jmt.get_root_hash_option(0);

    println!("root at version 0 after: {:?}", root);

    let mut user_vs_balances = HashMap::new();
    {
        let dave_balances = user_vs_balances.entry(k_dave).or_insert_with(|| HashMap::new());
        dave_balances.insert(btc, 200);
    }
    {
        let alice_balances = user_vs_balances.entry(k_alice).or_insert_with(|| HashMap::new());
        alice_balances.insert(btc, 400);
    }
    {
        let bob_balances = user_vs_balances.entry(k_bob).or_insert_with(|| HashMap::new());
        bob_balances.insert(btc, 500);
    }

    // 5) Apply updates to get Rf: ONLY alice changes to 150.
    let mut v1 = BTreeMap::new();
    for(key, value) in user_vs_balances.iter() {
        let v = map_to_bytes_le(value, false);
        v1.insert(key.clone(), v);
    }
    let root = jmt.get_root_hash_option(1);

    println!("root at version 1 before: {:?}", root);

    let (rf, _writes1) = jmt.put_value_set(v1.clone(), 1).expect("commit v1");


    store.write_node_batch(&_writes1.node_batch).expect("commit v1");

    let mut single_delta_hosts = Vec::new();

    for(key, value) in v1.iter() {
        let (val_old, proof) = jmt.get_with_proof(*key, 0).expect("proof A@R0");
        let (frames, l_bits) = frames_from_single_proof_r0::<Sha256>(*key, proof.siblings());
        proof.verify(r0, *key, val_old.clone()).expect("verification failed");

        match val_old {
            None => {
                let old_base_at_leaf_path = proof.leaf().map(|leaf| leaf.hash::<Sha256>());
                let conflicting_key       = proof.leaf().map(|leaf| leaf.key_hash());
                single_delta_hosts.push(SingleForDeltaHost {
                    key: *key,
                    old_value_hash: None,
                    new_value_hash: Some(value_hash(value.clone().unwrap().as_slice())),
                    frames_r0: frames,
                    leaf_path: encode_prefix_from_key(l_bits, key),
                    old_base_at_leaf_path,
                    conflicting_key,
                });
            }
            Some(val_old) => {
                single_delta_hosts.push(SingleForDeltaHost {
                    key: *key,
                    old_value_hash: Some(value_hash(val_old.as_slice())),
                    new_value_hash: Some(value_hash(value.clone().unwrap().as_slice())),
                    frames_r0: frames,
                    leaf_path: encode_prefix_from_key(l_bits, key),
                    old_base_at_leaf_path: None,
                    conflicting_key: None,
                })
            }
        }
    }

    let delta = build_delta_from_r0_single_proofs(single_delta_hosts);

    let root = jmt.get_root_hash_option(1).unwrap().unwrap();

    println!("root at version 1 after: {:?}", root);

    let ok = verify_delta_multiproof::<Sha256>(&Sha256::new(), r0, rf, &delta);
    println!("verify_delta_multiproof: {}", ok);
    assert!(ok, "delta multiproof should verify but did not");
}

#[test]
fn delta_multiproof_integration_fails_if_unrelated_key_changed() {
    // Build R0 with the same three accounts.
    let store = MockTreeStore::default();
    let jmt = JellyfishMerkleTree::<_, Sha256>::new(&store);

    let k_alice = KeyHash::with::<Sha256>(b"alice");
    let k_bob   = KeyHash::with::<Sha256>(b"bob");
    let k_carol = KeyHash::with::<Sha256>(b"carol");

    let v_alice_old = b"100";
    let v_bob_old = b"200";
    let v_carol_old = b"300";

    let v_alice_new = b"150";
    let v_bob_new = b"400";
    let v_carol_new = b"500";

    let mut v0 = BTreeMap::new();
    v0.insert(k_alice, Some(v_alice_old.to_vec()));
    v0.insert(k_bob,   Some(v_bob_old.to_vec()));
    v0.insert(k_carol, Some(v_carol_old.to_vec()));
    let (r0, tree_update_batch) = jmt.put_value_set(v0, 0).expect("commit v0");
    store.write_node_batch(&tree_update_batch.node_batch);

    // Single proofs for S = {alice, bob} under R0
    let (_va, proof_a_r0) = jmt.get_with_proof(k_alice, 0).expect("proof A@R0");
    let (_vb, proof_b_r0) = jmt.get_with_proof(k_bob,   0).expect("proof B@R0");

    let (frames_a, la_bits) = frames_from_single_proof_r0::<Sha256>(k_alice, proof_a_r0.siblings());
    let (frames_b, lb_bits) = frames_from_single_proof_r0::<Sha256>(k_bob,   proof_b_r0.siblings());

    let delta = build_delta_from_r0_single_proofs(vec![
        SingleForDeltaHost {
            key: k_alice,
            old_value_hash: Some(value_hash(v_alice_old)),
            new_value_hash: Some(value_hash(v_alice_new)), // intending to change only alice
            frames_r0: frames_a,
            leaf_path: encode_prefix_from_key(la_bits, &k_alice),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
        SingleForDeltaHost {
            key: k_bob,
            old_value_hash: Some(value_hash(v_bob_old)),
            new_value_hash: Some(value_hash(v_bob_new)),
            frames_r0: frames_b,
            leaf_path: encode_prefix_from_key(lb_bits, &k_bob),
            old_base_at_leaf_path: None,
            conflicting_key: None,
        },
    ]);

    // Commit Rf' where an *unrelated* key (carol) also changes.
    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec())); // intended change
    v1.insert(k_bob, Some(v_bob_new.to_vec())); // intended change
    v1.insert(k_carol, Some(v_carol_new.to_vec())); // malicious/unrelated change
    let (rf_bad, _) = jmt.put_value_set(v1, 1).expect("commit v1_bad");

    // The delta proof should FAIL, because we reused boundary siblings from R0
    // and they cannot produce rf_bad if Carol's subtree changed.
    assert!(
        !verify_delta_multiproof::<Sha256>(&Sha256::new(), r0, rf_bad, &delta),
        "delta multiproof should fail when unrelated key changed"
    );
}