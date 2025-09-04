//! Storage-backed integration test for the JMT delta multiproof.
//!
//! This test builds a JMT in memory, commits an initial state R0, fetches single
//! proofs under R0, converts those single proofs into `InternalFrame`s, builds a
//! single `DeltaMultiProof`, then commits a new state Rf and verifies that the
//! SAME internal frames reconstruct both R0 (with old values) and Rf (with new
//! values). It also demonstrates that changing an unrelated key causes the
//! verification to fail.

use std::collections::BTreeMap;
use sha2::Sha256;
use crate::{JellyfishMerkleTree, KeyHash, SimpleHasher};
use crate::mock::MockTreeStore;
use crate::multiproof::{ encode_prefix_from_key, frames_from_single_proof_r0, verify_delta_multiproof, DeltaLeaf, DeltaMultiProof, InternalFrame, Path};
use crate::writer::TreeWriter;

/// ---- Helpers to map single proofs -> InternalFrames (R0) --------------------


/// Convenience: compute JMT value hash the same way your crate does.
fn value_hash(v: &[u8]) -> [u8; 32] {
    Sha256::hash(v)
}

/// Host-side input for unioning multiple single proofs (all w.r.t. R0).
pub struct SingleForDeltaHost<'a> {
    pub key: KeyHash,
    pub old_value_hash: Option<[u8; 32]>, // None for insert
    pub new_value_hash: Option<[u8; 32]>, // None for delete
    pub frames_r0: &'a [InternalFrame],
    pub leaf_path: Path,
}

/// Union per-key frames into one DeltaMultiProof. No storage reads; duplicates are merged in verify.
pub fn build_delta_from_r0_single_proofs(
    inputs: Vec<SingleForDeltaHost<'_>>,
) -> DeltaMultiProof {
    let mut leaves: Vec<DeltaLeaf> = inputs.iter().map(|it| DeltaLeaf {
        key: it.key,
        leaf_path: it.leaf_path.clone(),
        old_value_hash: it.old_value_hash,
        new_value_hash: it.new_value_hash,
    }).collect();

    // deterministic order
    leaves.sort_unstable_by(|a, b| a.key.0.cmp(&b.key.0));

    // concatenate frames; verifier will normalize
    let mut internals: Vec<InternalFrame> = Vec::new();
    for it in inputs {
        internals.extend_from_slice(it.frames_r0);
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
            frames_r0: &frames_a,
            leaf_path: encode_prefix_from_key(la_bits, &k_alice),
        },
        SingleForDeltaHost {
            key: k_bob,
            old_value_hash: Some(value_hash(v_bob_old)),
            new_value_hash: Some(value_hash(v_bob_new)),
            frames_r0: &frames_b,
            leaf_path: encode_prefix_from_key(lb_bits, &k_bob),
        },
    ]);

    // 5) Apply updates to get Rf: ONLY alice changes to 150.
    let mut v1 = BTreeMap::new();
    v1.insert(k_alice, Some(v_alice_new.to_vec()));
    v1.insert(k_bob, Some(v_bob_new.to_vec()));
    let (rf, _writes1) = jmt.put_value_set(v1, 1).expect("commit v1");

    // 6) Verify the delta multiproof (should pass).
    assert!(
        verify_delta_multiproof::<Sha256>(&Sha256::new(), r0, rf, &delta),
        "delta multiproof should verify but did not"
    );
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

    print!("Root: {:?}", r0);
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
            frames_r0: &frames_a,
            leaf_path: encode_prefix_from_key(la_bits, &k_alice),
        },
        SingleForDeltaHost {
            key: k_bob,
            old_value_hash: Some(value_hash(v_bob_old)),
            new_value_hash: Some(value_hash(v_bob_new)),
            frames_r0: &frames_b,
            leaf_path: encode_prefix_from_key(lb_bits, &k_bob),
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