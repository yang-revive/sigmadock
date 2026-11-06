"""The per-fragment SE(3) target: reachable, optimal, and not arbitrary.

Three properties are what the target has to have, and they are checked here independently of how
``rigid`` arrives at them: the pose it names is one a rigid template can actually take, it is the
closest such pose, and the freedom left over -- symmetry relabellings and the gauge circle of a
collinear fragment -- is resolved by nearness rather than by whatever the SVD happened to return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conftest import DUMMY
from pocketdit.chem import chirality_reference, load_ligand, signed_volume
from pocketdit.fragment import fragment, is_collinear
from pocketdit.rigid import (
    MAX_LIGAND_ATOMS,
    TIE_TOL,
    apply_transforms,
    embed_template,
    fragment_relabellings,
    fragment_targets,
    free_axis,
    is_template_symmetry,
    joint_errors,
    kabsch,
    load_template,
    pin_gauge,
    rotation_about,
    rotation_angle,
    rotation_distance,
    statistics,
    symmetry_classes,
)

#: Molecules whose fragments carry a genuine symmetry: a mono-substituted ring (the two ortho and
#: the two meta carbons), three equilateral methyls, two methyls on one nitrogen.
SYMMETRIC = [
    "CCc1ccccc1",
    "CN(C)c1ccccc1",
    "CC(C)(C)c1ccc(OC)cc1",
    "c1ccc(-c2ccccc2)cc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


def embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0, smiles
    return Chem.RemoveHs(mol)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q[:, ::-1]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


@pytest.fixture(scope="module")
def ligands() -> list[tuple[str, Chem.Mol]]:
    if not DUMMY.exists():
        pytest.skip(f"example data not found at {DUMMY}")
    return [(p.parent.name, load_ligand(p)) for p in sorted(DUMMY.glob("*/*_ligand.sdf"))]


@pytest.fixture(scope="module")
def cases(ligands: list[tuple[str, Chem.Mol]]) -> list[tuple[str, Chem.Mol, np.ndarray, np.ndarray]]:
    """Each example ligand with its true pose and the start conformer shipped beside it.

    That conformer is the internal geometry inference is actually handed, so it is the setting the
    target has to be right in -- not the crystal pose, where there is nothing left to fit.
    """
    out = []
    for name, mol in ligands:
        start = next(iter(sorted((DUMMY / name).glob("*_ligand_start_conf.sdf"))), None)
        template = load_template(mol, start) if start else embed_template(mol)
        out.append((name, mol, mol.GetConformer().GetPositions(), template))
    return out


# --------------------------------------------------------------------------------------
# Rigid fits
# --------------------------------------------------------------------------------------
def test_kabsch_recovers_a_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    p = rng.normal(size=(7, 3))
    rot, trans = random_rotation(rng), rng.normal(size=3) * 5
    r, t = kabsch(p, p @ rot.T + trans)
    assert np.allclose(r, rot, atol=1e-9)
    assert np.allclose(t, trans, atol=1e-9)


def test_kabsch_never_reflects() -> None:
    """A mirrored fragment is a different molecule, so a reflection is not on offer at any cost."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=(6, 3))
    mirrored = p * np.array([1.0, 1.0, -1.0])
    r, t = kabsch(p, mirrored)
    assert np.linalg.det(r) > 0
    assert rms(p @ r.T + t, mirrored) > 0.1


def test_apply_transforms_inverts_the_target(cases) -> None:
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        assert np.allclose(apply_transforms(frag.frag_idx, template, tgt.rot, tgt.trans, tgt.center), tgt.pos), name
        # The centre is the template centroid, so leaving it out has to make no difference.
        assert np.allclose(apply_transforms(frag.frag_idx, template, tgt.rot, tgt.trans), tgt.pos), name


# --------------------------------------------------------------------------------------
# The aligned template is the target
# --------------------------------------------------------------------------------------
def test_the_target_keeps_the_template_internal_geometry(cases) -> None:
    """Every fragment of the target is the template fragment, bond for bond and angle for angle.

    This is what "rigid" means, and it is the reason the target cannot be the true pose: the true
    pose's internal geometry is a different one.
    """
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        for f in range(frag.num_fragments):
            atoms = np.flatnonzero(frag.frag_idx == f)
            inner = np.linalg.norm(template[atoms][:, None] - template[atoms][None], axis=-1)
            placed = np.linalg.norm(tgt.pos[atoms][:, None] - tgt.pos[atoms][None], axis=-1)
            assert np.allclose(inner, placed, atol=1e-8), f"{name}: fragment {f} was deformed"


def test_a_template_that_is_the_true_pose_is_hit_exactly(cases) -> None:
    """Re-docking with the crystal conformer is the degenerate case: nothing is left to fit."""
    for name, mol, true, _ in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, true, max_atoms=None)
        assert tgt.rmsd.max() < 1e-8, name
        assert np.allclose(tgt.pos, true, atol=1e-8), name
        assert np.array_equal(tgt.pairing, np.arange(mol.GetNumAtoms())), name


def test_embed_template_keeps_the_order_and_the_enantiomer() -> None:
    """A mirrored template would be fitted by a rotation that cannot reach it, since ``kabsch``
    refuses reflections -- so the embedding has to come back on the same hand it went in on."""
    mol = embed("C[C@@H](O)[C@H](N)C(=O)O")
    template = embed_template(mol)
    assert template.shape == (mol.GetNumAtoms(), 3)
    nbrs, sign = chirality_reference(mol, mol.GetConformer().GetPositions())
    centres = np.flatnonzero(sign)
    assert len(centres) == 2
    for i in centres:
        volume = signed_volume(template[i], *(template[j] for j in nbrs[i]))
        assert np.sign(volume) == sign[i], f"stereocentre {i} was mirrored"


def test_a_separate_conformer_leaves_a_floor(cases) -> None:
    """With the geometry inference really has, the target is *not* the true pose -- and the gap is
    the price of the parameterisation, which the target makes visible instead of burying it."""
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        floor = tgt.total_rmsd(true)
        assert floor > 1e-3, f"{name}: an ETKDG template should not reproduce the crystal pose"
        assert floor < 0.5, f"{name}: floor of {floor:.3f} Å is too high to be a rigid fit"
        # And it beats placing the whole ligand as one rigid body, which is what the cuts buy.
        whole_rot, whole_trans = kabsch(template, true)
        assert floor < rms(template @ whole_rot.T + whole_trans, true), name


def test_the_target_is_the_best_rigid_placement(cases) -> None:
    """Perturbing a fragment's transform can only make it worse: the fit is at the optimum."""
    rng = np.random.default_rng(2)
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        for f in range(frag.num_fragments):
            atoms = np.flatnonzero(frag.frag_idx == f)
            matched = true[tgt.pairing[atoms]]
            for _ in range(4):
                rot = tgt.rot[f] @ rotation_about(rng.normal(size=3), float(rng.uniform(0.02, 0.2)))
                trans = tgt.trans[f] + rng.normal(size=3) * 0.1
                moved = (template[atoms] - tgt.center[f]) @ rot.T + trans
                assert rms(moved, matched) >= tgt.rmsd[f] - 1e-9, f"{name}: fragment {f} is not optimal"


def test_the_fit_is_equivariant_when_the_reference_moves_with_the_pose(cases) -> None:
    """The fit itself is equivariant; which equivalent solution is taken is decided against the
    reference, so the reference has to be carried along for the whole target to transform."""
    rng = np.random.default_rng(3)
    rot, shift = random_rotation(rng), rng.normal(size=3) * 7
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        moved = fragment_targets(mol, frag, true @ rot.T + shift, template, reference=rot, max_atoms=None)
        assert np.allclose(moved.pos, tgt.pos @ rot.T + shift, atol=1e-7), name
        assert np.allclose(moved.rot, rot @ tgt.rot, atol=1e-7), name
        assert np.allclose(moved.trans, tgt.trans @ rot.T + shift, atol=1e-7), name
        assert np.allclose(moved.rmsd, tgt.rmsd, atol=1e-8), name
        assert np.array_equal(moved.pairing, tgt.pairing), name


def test_the_target_is_deterministic(cases) -> None:
    for name, mol, true, template in cases:
        frag = fragment(mol)
        first = fragment_targets(mol, frag, true, template, max_atoms=None)
        again = fragment_targets(mol, frag, true, template, max_atoms=None)
        assert np.array_equal(first.rot, again.rot), name
        assert np.array_equal(first.pairing, again.pairing), name


# --------------------------------------------------------------------------------------
# Gauge: the rotation a collinear fragment leaves free
# --------------------------------------------------------------------------------------
def test_free_axis_reports_the_line_of_a_collinear_fragment() -> None:
    assert free_axis(np.array([[0.0, 0, 0], [1.5, 0, 0]])) is not None
    assert abs(abs(free_axis(np.array([[0.0, 0, 0], [1.5, 0, 0]]))[0]) - 1) < 1e-9
    assert free_axis(np.array([[0.0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]])) is not None  # a nitrile
    assert free_axis(np.array([[0.0, 0, 0], [1.5, 0, 0], [0.7, 1.3, 0]])) is None
    assert free_axis(np.zeros((1, 3))) is None  # a single atom leaves everything free, not an axis


def test_pinning_the_gauge_does_not_move_any_atom() -> None:
    """It replaces the rotation with another one that produces the same coordinates. That is what
    makes it a gauge choice: there is nothing to fit, only a representative to agree on."""
    rng = np.random.default_rng(4)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    template = np.stack([-0.74 * axis, 0.74 * axis])
    rot = random_rotation(rng)
    pinned, free = pin_gauge(rot, template, np.eye(3))
    assert free
    assert np.allclose(template @ pinned.T, template @ rot.T, atol=1e-9)
    assert rotation_angle(pinned) <= rotation_angle(rot) + 1e-9


def test_the_pinned_rotation_is_the_smallest_one_that_works() -> None:
    """For a two-atom fragment that is the rotation about ``a x b`` by the angle between them."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 1.0]) / math.sqrt(2)
    template, target = np.stack([-0.75 * a, 0.75 * a]), np.stack([-0.75 * b, 0.75 * b])
    rot, trans = kabsch(template, target)
    pinned, free = pin_gauge(rot, template, np.eye(3))
    assert free
    assert np.allclose(pinned, rotation_about(np.cross(a, b), math.acos(float(a @ b))), atol=1e-8)
    assert np.allclose(template @ pinned.T + trans, target, atol=1e-8)


def test_every_member_of_the_gauge_family_pins_to_the_same_representative() -> None:
    """The bare SVD returns an arbitrary member on a rank-deficient fragment, so without this the
    rotation target would not be reproducible across machines, let alone continuous."""
    rng = np.random.default_rng(5)
    axis = np.array([0.3, -0.7, 0.6])
    axis /= np.linalg.norm(axis)
    template = np.stack([-0.7 * axis, 0.7 * axis])
    rot = random_rotation(rng)
    reference = random_rotation(rng)
    pinned, _ = pin_gauge(rot, template, reference)
    for phi in (0.3, 1.1, -2.5, 3.0):
        again, _ = pin_gauge(rot @ rotation_about(axis, phi), template, reference)
        assert np.allclose(again, pinned, atol=1e-8)
    assert (
        rotation_distance(reference, pinned)
        <= min(
            rotation_distance(reference, rot @ rotation_about(axis, phi)) for phi in np.linspace(-math.pi, math.pi, 73)
        )
        + 1e-8
    )


def test_a_single_atom_fragment_takes_the_reference_outright() -> None:
    rng = np.random.default_rng(6)
    reference = random_rotation(rng)
    pinned, free = pin_gauge(random_rotation(rng), np.zeros((1, 3)), reference)
    assert free and np.allclose(pinned, reference)


def test_a_lone_atom_fragment_is_placed_without_a_rotation() -> None:
    """``selection="none"`` is the one setting that leaves single atoms, and they have no
    orientation at all: the translation is the whole answer."""
    mol = embed("CCCCC")
    frag = fragment(mol, selection="none")
    true, template = mol.GetConformer().GetPositions(), embed_template(mol)
    tgt = fragment_targets(mol, frag, true, template)
    lone = [f for f in range(frag.num_fragments) if int((frag.frag_idx == f).sum()) == 1]
    assert lone, "pentane cut at both torsions should isolate its middle carbon"
    for f in lone:
        assert np.allclose(tgt.rot[f], np.eye(3))
        assert tgt.gauge[f]
        assert tgt.rmsd[f] < 1e-9
        assert np.allclose(tgt.trans[f], true[frag.frag_idx == f][0])


def test_the_gauge_flag_matches_collinearity(cases) -> None:
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        for f in range(frag.num_fragments):
            atoms = np.flatnonzero(frag.frag_idx == f)
            assert bool(tgt.gauge[f]) == is_collinear(template[atoms]), f"{name}: fragment {f}"


# --------------------------------------------------------------------------------------
# Symmetry: the relabellings that describe the same pose
# --------------------------------------------------------------------------------------
def relabellings_of(mol: Chem.Mol, frag, f: int) -> list[np.ndarray]:
    atoms = np.flatnonzero(frag.frag_idx == f)
    return fragment_relabellings(mol, atoms, symmetry_classes(mol))


def fragment_of_size(frag, size: int) -> int:
    return next(f for f in range(frag.num_fragments) if int((frag.frag_idx == f).sum()) == size)


def test_a_monosubstituted_ring_has_one_twofold_relabelling() -> None:
    """Its ortho pair and its meta pair swap; the ipso and para carbons stay, so the 180° flip
    about the ipso-para axis is the only symmetry, not the ring's full sixfold."""
    mol = embed("CCc1ccccc1")
    frag = fragment(mol)
    perms = relabellings_of(mol, frag, fragment_of_size(frag, 6))
    assert len(perms) == 2
    assert np.array_equal(perms[0], np.arange(6))  # the identity is always offered
    assert sorted(np.bincount(perms[1], minlength=6).tolist()) == [1] * 6  # and it is a permutation
    assert int((perms[1] == np.arange(6)).sum()) == 2  # ipso and para are the fixed points
    pos = mol.GetConformer().GetPositions()
    atoms = np.flatnonzero(frag.frag_idx == fragment_of_size(frag, 6))
    centred = pos[atoms] - pos[atoms].mean(0)
    assert all(is_template_symmetry(centred, p) for p in perms)  # both are rotations of the ring


def test_tert_butyl_offers_its_mirror_relabellings_too() -> None:
    """All six permutations of three equivalent methyls are chemically valid relabellings, and
    only three of them are rotations of the fragment. The mirror ones have to stay on offer: a
    tert-butyl is achiral but its *labelling* is chiral, so two conformers of the same molecule
    routinely disagree about which way round the labels run, and then the identity labelling is
    the one that cannot be fitted."""
    mol = embed("CC(C)(C)c1ccc(OC)cc1")
    frag = fragment(mol)
    tbutyl = fragment_of_size(frag, 4)
    atoms = np.flatnonzero(frag.frag_idx == tbutyl)
    perms = relabellings_of(mol, frag, tbutyl)
    assert len(perms) == 6
    template = embed_template(mol)
    centred = template[atoms] - template[atoms].mean(0)
    assert sum(is_template_symmetry(centred, p) for p in perms) == 3

    # On this molecule the mirror relabelling is what rescues the fit, by a factor of about ten.
    true = embed_template(mol, seed=1)
    tgt = fragment_targets(mol, frag, true, template)
    identity, shift = kabsch(centred, true[atoms])
    assert rms(centred @ identity.T + shift, true[atoms]) > 0.3
    assert tgt.rmsd[tbutyl] < 0.1
    assert tgt.relabelled[tbutyl]


def test_an_atom_bonded_across_a_cut_is_never_relabelled(cases) -> None:
    """It anchors a neighbouring fragment that was fitted on its own, so moving it would leave the
    assembled target with a stretched joint."""
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        for bond_idx in frag.cut_bonds:
            bond = mol.GetBondWithIdx(int(bond_idx))
            for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                assert tgt.pairing[atom] == atom, f"{name}: cut atom {atom} was relabelled"


def test_a_rotational_relabelling_describes_the_same_pose() -> None:
    """A relabelling that is a rotation of the template produces the same occupied positions, and
    so fits exactly as well. That is what makes choosing between those free -- and it is why they,
    unlike the mirror relabellings, have to be settled by nearness rather than by the fit."""
    seen = 0
    for smiles in SYMMETRIC:
        mol = embed(smiles)
        frag = fragment(mol)
        true, template = mol.GetConformer().GetPositions(), embed_template(mol)
        for f in range(frag.num_fragments):
            atoms = np.flatnonzero(frag.frag_idx == f)
            centred = template[atoms] - template[atoms].mean(0)
            perms = [p for p in relabellings_of(mol, frag, f) if is_template_symmetry(centred, p)]
            if len(perms) == 1:
                continue
            seen += 1
            poses, fits = [], []
            for perm in perms:
                rot, trans = kabsch(centred, true[atoms[perm]])
                poses.append(centred @ rot.T + trans)
                fits.append(rms(poses[-1], true[atoms[perm]]))
            assert max(fits) - min(fits) < TIE_TOL, f"{smiles}: fragment {f} fits {fits}"
            for other in poses[1:]:
                gap = np.linalg.norm(poses[0][:, None] - other[None], axis=-1).min(axis=1).max()
                assert gap < 0.2, f"{smiles}: fragment {f} moved by {gap:.3f} Å"
    assert seen, "no fragment with a rotational symmetry in the corpus to check"


def test_the_nearest_tied_solution_is_the_one_taken() -> None:
    """Of the relabellings that fit as well as the best, none may be nearer the reference than the
    one the target uses."""
    rng = np.random.default_rng(7)
    for smiles in SYMMETRIC:
        mol = embed(smiles)
        frag = fragment(mol)
        true, template = mol.GetConformer().GetPositions(), embed_template(mol)
        for reference in (np.eye(3), random_rotation(rng)):
            tgt = fragment_targets(mol, frag, true, template, reference=reference)
            for f in range(frag.num_fragments):
                atoms = np.flatnonzero(frag.frag_idx == f)
                centred = template[atoms] - template[atoms].mean(0)
                offered = []
                for perm in relabellings_of(mol, frag, f):
                    rot, trans = kabsch(centred, true[atoms[perm]])
                    rot, _ = pin_gauge(rot, centred, reference)
                    offered.append((rms(centred @ rot.T + trans, true[atoms[perm]]), rot))
                best = min(fit for fit, _ in offered)
                tied = [rotation_distance(reference, rot) for fit, rot in offered if fit <= best + TIE_TOL]
                assert tgt.rmsd[f] <= best + TIE_TOL, f"{smiles}: fragment {f} took a worse fit"
                assert rotation_distance(reference, tgt.rot[f]) <= min(tied) + 1e-8, f"{smiles}: fragment {f}"


def test_the_reference_selects_which_tied_solution_is_used() -> None:
    """Point the reference at one of the ring's two rotational labellings and that one comes back."""
    mol = embed("CCc1ccccc1")
    frag = fragment(mol)
    true, template = mol.GetConformer().GetPositions(), embed_template(mol)
    ring = fragment_of_size(frag, 6)
    atoms = np.flatnonzero(frag.frag_idx == ring)
    centred = template[atoms] - template[atoms].mean(0)
    perms = relabellings_of(mol, frag, ring)
    assert len(perms) == 2
    for perm in perms:
        wanted, _ = kabsch(centred, true[atoms[perm]])
        tgt = fragment_targets(mol, frag, true, template, reference=wanted)
        assert np.allclose(tgt.rot[ring], wanted, atol=1e-7)
        assert np.array_equal(tgt.pairing[atoms], atoms[perm])
        assert tgt.relabelled[ring] == bool((perm != np.arange(len(atoms))).any())


# --------------------------------------------------------------------------------------
# Joints between fragments
# --------------------------------------------------------------------------------------
def test_cut_bonds_stay_near_bonding_distance(cases) -> None:
    """Nothing in the per-fragment fits holds a cut bond together, so this is the one error the
    target does not control. It has to stay small all the same, or the target is not a molecule."""
    for name, mol, true, template in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, template, max_atoms=None)
        errors = joint_errors(mol, frag, tgt.pos, true)
        if len(errors):
            assert errors.max() < 0.75, f"{name}: a joint is off by {errors.max():.2f} Å"
    # A fit against the true pose itself has nothing to drift, joints included.
    for name, mol, true, _ in cases:
        frag = fragment(mol)
        tgt = fragment_targets(mol, frag, true, true, max_atoms=None)
        errors = joint_errors(mol, frag, tgt.pos, true)
        assert not len(errors) or errors.max() < 1e-8, name


# --------------------------------------------------------------------------------------
# The atom cap
# --------------------------------------------------------------------------------------
def test_a_ligand_over_the_atom_cap_is_refused() -> None:
    """Past 50 heavy atoms the ligands are peptides and macrocycles whose torsion counts are
    already past what the model is sized for, so no target is computed for them at all."""
    mol = Chem.MolFromSmiles("C" * (MAX_LIGAND_ATOMS + 1))
    zeros = np.zeros((mol.GetNumAtoms(), 3))
    with pytest.raises(ValueError, match="out of scope"):
        fragment_targets(mol, fragment(mol, selection="none"), zeros, zeros)


def test_the_cap_binds_exactly_at_the_atom_count() -> None:
    mol = embed("CCc1ccccc1")
    frag, true = fragment(mol), mol.GetConformer().GetPositions()
    n = mol.GetNumAtoms()
    assert fragment_targets(mol, frag, true, true, max_atoms=n).num_fragments == 2
    assert fragment_targets(mol, frag, true, true, max_atoms=None).num_fragments == 2
    with pytest.raises(ValueError, match="out of scope"):
        fragment_targets(mol, frag, true, true, max_atoms=n - 1)


def test_statistics_counts_what_the_cap_removed(ligands) -> None:
    paths = sorted(DUMMY.glob("*/*_ligand.sdf"))
    stats = statistics(paths, template="crystal")
    over = sum(1 for _, mol in ligands if mol.GetNumAtoms() > MAX_LIGAND_ATOMS)
    assert stats["filtered"] == over
    assert len(stats["rows"]) + over + len(stats["failures"]) == len(paths)
    assert max(r["total_rmsd"] for r in stats["rows"]) < 1e-8  # the crystal template fits itself
    assert statistics(paths, template="crystal", max_atoms=None)["filtered"] == 0
