"""The SE(3) target of every rigid fragment.

``fragment.py`` decides where to cut; this module says where each piece has to go. In fragment
mode a pose is not 3N coordinates but one rigid transform per fragment: the internal geometry of
a fragment is the input conformer's and is never touched, so the only freedom left is
``(R_f, t_f)`` in SE(3). Three decisions make up the target.

**The target is the aligned template, not the true pose.** The internal geometry of a fragment in
the input conformer -- ETKDG's at inference time -- is not the crystal pose's: bond lengths differ
by a few hundredths of an Å and angles by a degree or two. No rigid placement of the template
reproduces the crystal coordinates exactly, so regressing towards them aims the model at a pose it
cannot reach and leaves a floor in the loss that no amount of training removes. The target is
therefore the rigid placement that comes closest -- the Kabsch fit of the template fragment onto
the true one -- which makes the optimum attainable, and turns the residual into a number that can
be measured and reported (``FragmentTargets.rmsd``) instead of a bias hidden inside the loss.

**Equivalent solutions, and taking the nearest of the tied ones.** The fit is not unique, in two
different ways, and both have to be resolved or the target is arbitrary.

*Symmetry.* If the molecule cannot tell two of a fragment's atoms apart -- the ortho pair of a
mono-substituted ring, two methyls on one nitrogen -- then exchanging them is a second labelling of
the same chemical object, so fitting the template onto the exchanged true pose is a second
candidate for the transform. Every such relabelling is enumerated and fitted and the best fit wins.
The one thing they all have to respect is the atoms bonded across a cut: those anchor the
neighbouring fragments, which are fitted independently, so moving one would leave the assembled
target with a stretched joint.

Two kinds come out of that and they behave differently. A relabelling that is a *rotation* of the
template -- the 180° flip of a mono-substituted ring about its ipso-para axis -- necessarily fits
exactly as well as the identity, because relabelling and rotating are then the same operation.
Those are the ties, and settling them is what "nearest" is for. A relabelling that is only a
*mirror* of the template is a different pose and usually fits very differently, which is precisely
why it must not be filtered out: a tert-butyl's three methyls are equivalent, so its labels run
clockwise or anticlockwise around its own axis, and two conformers of one molecule routinely
disagree about which. Between two ETKDG conformers of 4-tert-butylanisole, the identity labelling
fits that tert-butyl at 0.45 Å and a mirror relabelling at 0.05 Å, which takes the whole molecule's
target from 0.26 Å to 0.03 Å. The fragment is achiral; only its labelling is not. Insisting on the
identity would ask the model for a pose 0.4 Å off for no reason at all.

*Gauge.* A two-atom or collinear fragment is unchanged by rotation about its own axis, so a whole
circle of rotations yields identical coordinates -- the spare degree of freedom
``fragment.is_collinear`` reports. Kabsch cannot pick between them: the covariance matrix is rank
deficient and the SVD returns whichever member the numerics land on, which is reproducible neither
across machines nor across small perturbations of the input. A single-atom fragment has all of
SO(3) free. The free component is pinned to the rotation nearest a reference (the identity, i.e.
the template's own frame, unless one is given), which costs nothing in coordinates and makes the
rotation target a well-defined, continuous function of the pose.

So the objective is the fit and nearness is the tie-break, with ``tie_tol`` setting how close
counts as tied. The rotational symmetries tie to floating-point precision, and the gauge circle is
flat by construction, so between them nearness decides everything that the fit genuinely cannot
see, and nothing it can. ``pairing`` records which true atom each target atom was fitted onto, so
a per-atom loss stays consistent with whichever labelling was chosen.

``reference`` is also the hook for the flow this target is for. During training the state is a
current ``R_t`` per fragment, and the geodesic the flow should follow goes to the *nearest*
equivalent target, not to a fixed representative -- otherwise a symmetric fragment is sent the
long way round its own symmetry group. Passing ``reference=R_t`` does that. The identity default
is the same rule against the template's own frame: the smallest turn that produces the pose.

**Ligands above 50 heavy atoms are out of scope.** The full-PDBBind sweep in the README splits
sharply at that size: at or below it (87.9% of the set) a ligand averages 5.2 fragments, 6.5
torsions and 19.0% gauge-degenerate fragments, while above it the averages run to 19-34 fragments,
33-61 torsions and 29-35% degeneracy. Those are peptides and macrocycles; their torsion counts
alone (p90 = 24, max 110) put them past the ``--max_torsions 20`` the model is sized for, and
every fragment target computed for them is work spent on a complex that cannot be trained on.
``MAX_LIGAND_ATOMS`` is the cap, and it is a filter on the ligand rather than on its fragments --
dropping the awkward fragments of a large ligand would leave a pose that cannot be assembled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from pocketdit.fragment import COLLINEAR_TOL, Fragmentation, fragment, is_collinear

MAX_LIGAND_ATOMS = 50  # heavy atoms; above this a ligand gets no fragment targets at all
SYMMETRY_TOL = 0.1  # Å, how closely a relabelling has to reproduce the template to be a rotation of it
# Å, residuals this close count as tied, and the tie goes to the solution nearest the reference.
# The rotational symmetries tie exactly, so this only has to cover how far a real fragment is from
# being exactly symmetric -- a distorted ring, a conformer-dependent bond angle.
TIE_TOL = 0.1
MAX_RELABELLINGS = 64  # cap on the enumeration; beyond it the extra solutions are not worth the search


@dataclass(frozen=True)
class FragmentTargets:
    """Where every rigid fragment has to go.

    ``pos`` is the pose the transforms produce, i.e. the aligned template: fragment internal
    geometry exactly as in the template, placed as well as a rigid body can be placed. It is the
    regression target. ``rmsd`` is what the parameterisation costs, per fragment, measured against
    the true pose under the chosen relabelling.
    """

    rot: np.ndarray  # [F, 3, 3] float64, rotation of each fragment about its own template centroid
    trans: np.ndarray  # [F, 3] where that centroid goes, i.e. the fragment's target centroid
    center: np.ndarray  # [F, 3] the template centroid the rotation acts about
    pos: np.ndarray  # [n, 3] the aligned-template pose the transforms produce
    rmsd: np.ndarray  # [F] per-fragment fit residual against the true pose (Å)
    pairing: np.ndarray  # [n] which true atom each target atom was fitted onto (identity if none)
    gauge: np.ndarray  # [F] bool, the rotation had a free component that had to be pinned
    relabelled: np.ndarray  # [F] bool, a non-identity relabelling won the fit
    num_relabellings: np.ndarray  # [F] how many relabellings were fitted, the identity included
    num_symmetries: np.ndarray  # [F] how many of those are rotations of the template, i.e. exact ties

    @property
    def num_fragments(self) -> int:
        return len(self.rot)

    def total_rmsd(self, target_pos: np.ndarray) -> float:
        """RMSD of the whole aligned-template pose against the true one: the fragment-mode floor."""
        true = np.asarray(target_pos, dtype=np.float64)[self.pairing]
        return float(np.sqrt(((self.pos - true) ** 2).sum(-1).mean()))


# --------------------------------------------------------------------------------------
# Rigid transforms
# --------------------------------------------------------------------------------------
def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation minimising ``|p @ R.T + t - q|``, over proper rotations only.

    Reflections are excluded even where one would fit better: a mirrored fragment is a different
    molecule, and for a stereocentre-bearing fragment it is the wrong enantiomer. The row-vector
    convention (``x @ R.T``) is the one ``data.perturb_reference`` already uses.
    """
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    flip = np.ones(3)
    flip[2] = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = (vt.T * flip) @ u.T
    return rot, qc - pc @ rot.T


def apply_transforms(
    frag_idx: np.ndarray,
    template_pos: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Place each fragment's template geometry by its own transform: ``[n, 3]``.

    The inverse of ``fragment_targets``' output, and the decoder a fragment-parameterised model
    needs at sampling time.
    """
    template_pos = np.asarray(template_pos, dtype=np.float64)
    out = np.empty_like(template_pos)
    for f in range(len(rot)):
        atoms = np.flatnonzero(frag_idx == f)
        c = template_pos[atoms].mean(0) if center is None else center[f]
        out[atoms] = (template_pos[atoms] - c) @ rot[f].T + trans[f]
    return out


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation by ``angle`` radians about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def rotation_angle(rot: np.ndarray) -> float:
    """Geodesic distance from the identity, in radians: the amount of turning ``rot`` is."""
    return float(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)))


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic distance between two rotations, in radians."""
    return rotation_angle(a.T @ b)


# --------------------------------------------------------------------------------------
# Gauge: the rotation a degenerate fragment leaves free
# --------------------------------------------------------------------------------------
def free_axis(pos: np.ndarray, tol: float = COLLINEAR_TOL) -> np.ndarray | None:
    """Axis a rotation about which leaves ``pos`` where it is, or ``None`` if there is none.

    The line direction of a collinear fragment (two atoms, a nitrile). A single atom, or several
    atoms on top of each other, has no axis to report because *every* rotation is free; that case
    is handled by ``pin_gauge`` taking the reference outright.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 2 or not is_collinear(pos, tol):
        return None
    centred = pos - pos.mean(0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    if singular[0] < tol:
        return None  # coincident atoms: nothing is free about an axis, everything is
    return vt[0]


def pin_gauge(
    rot: np.ndarray, template: np.ndarray, reference: np.ndarray, tol: float = COLLINEAR_TOL
) -> tuple[np.ndarray, bool]:
    """Replace ``rot`` by the member of its gauge family nearest ``reference``.

    A collinear fragment's family is ``rot @ R_axis(phi)``; the ``phi`` closest to ``reference``
    has a closed form, since the geodesic distance is set by
    ``tr(Q R_axis(phi)) = c + A sin(phi) - B cos(phi)`` with ``Q = reference.T @ rot``, maximal at
    ``phi = atan2(A, -B)``. For a single atom the family is all of SO(3) and the answer is
    ``reference`` itself. The coordinates the transform produces are unchanged either way -- that
    is what makes this a gauge choice and not a fit.

    ``template`` has to be centred on the point the rotation acts about, since that is what puts
    the free axis through the origin and leaves the fragment's own atoms fixed under it.
    """
    template = np.asarray(template, dtype=np.float64)
    if len(template) == 1 or np.allclose(template, template.mean(0), atol=tol):
        return np.asarray(reference, dtype=np.float64).copy(), True
    axis = free_axis(template, tol)
    if axis is None:
        return rot, False
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    q = np.asarray(reference, dtype=np.float64).T @ rot
    return rot @ rotation_about(a, math.atan2(float(np.trace(q @ k)), -float(np.trace(q @ k @ k)))), True


# --------------------------------------------------------------------------------------
# Symmetry: the relabellings that describe the same pose
# --------------------------------------------------------------------------------------
def symmetry_classes(mol: Chem.Mol) -> np.ndarray:
    """Canonical rank per atom with ties kept: equal ranks mean the molecule cannot tell apart."""
    return np.asarray(Chem.CanonicalRankAtoms(mol, breakTies=False), dtype=np.int64)


def _automorphisms(classes: np.ndarray, bonds: np.ndarray, limit: int) -> list[np.ndarray]:
    """Permutations that preserve every atom's class and every bond inside the fragment.

    The identity is always tried first, so it survives the cap, and the bond check prunes hard
    enough that the search is negligible at the sizes fragments come in.
    """
    k = len(classes)
    options = []
    for i in range(k):
        same = np.flatnonzero(classes == classes[i])
        options.append(np.concatenate(([i], same[same != i])))  # identity first, so it always survives the cap
    if all(len(o) == 1 for o in options):
        return [np.arange(k)]

    out: list[np.ndarray] = []
    perm = np.zeros(k, dtype=np.int64)
    used = np.zeros(k, dtype=bool)

    def walk(i: int) -> bool:
        if i == k:
            out.append(perm.copy())
            return len(out) < limit
        for j in options[i]:
            if used[j] or any(bonds[j, perm[m]] != bonds[i, m] for m in range(i)):
                continue
            perm[i], used[j] = j, True
            keep_going = walk(i + 1)
            used[j] = False
            if not keep_going:
                return False
        return True

    walk(0)
    return out


def fragment_relabellings(
    mol: Chem.Mol,
    atoms: np.ndarray,
    classes: np.ndarray | None = None,
    limit: int = MAX_RELABELLINGS,
) -> list[np.ndarray]:
    """Relabellings of one fragment's atoms, as permutations of ``atoms``' own order.

    A relabelling qualifies when it maps every atom onto one the molecule cannot tell apart and
    keeps every bond inside the fragment -- an automorphism of the fragment as a chemical object.
    Geometry is deliberately not consulted: whether a relabelling happens to be a rotation of the
    template, a mirror of it, or neither is exactly what the fit is there to measure, and filtering
    on it would throw away the mirror relabellings a fragment such as a tert-butyl needs.

    An atom bonded across a cut is given a class of its own, which forces it to be a fixed point.
    It is where a neighbouring fragment attaches, and neighbours are fitted independently, so
    moving it would leave the assembled target with a stretched bond across the joint. The useful
    side effect is that a cut bond's two atoms are never relabelled, so the joint lengths of the
    target can be compared with the true ones atom for atom.
    """
    cls = (symmetry_classes(mol) if classes is None else classes)[atoms].copy()
    spare = int(cls.max()) + 1
    local = {int(a): i for i, a in enumerate(atoms)}
    bonds = np.zeros((len(atoms), len(atoms)), dtype=np.int64)
    for i, atom in enumerate(atoms):
        cut = False
        for nbr in mol.GetAtomWithIdx(int(atom)).GetNeighbors():
            j = local.get(nbr.GetIdx())
            if j is None:
                cut = True
            else:
                bonds[i, j] = int(mol.GetBondBetweenAtoms(int(atom), nbr.GetIdx()).GetBondType())
        if cut:  # bonded across a cut, so pin it with a class of its own
            cls[i] = spare
            spare += 1
    return _automorphisms(cls, bonds, limit)


def is_template_symmetry(centred: np.ndarray, perm: np.ndarray, tol: float = SYMMETRY_TOL) -> bool:
    """Whether a relabelling is a *rotation* of the template, rather than a mirror or neither.

    These are the relabellings that cannot be told apart by the fit: a rotation of the template
    produces the same occupied positions, so the relabelled fit and the identity fit have the same
    residual, necessarily. They are therefore the ties that ``reference`` settles, and counting
    them is how ``report`` separates real symmetry from the mirror relabellings.
    """
    rot, _ = kabsch(centred, centred[perm])
    return bool(np.sqrt(((centred @ rot.T - centred[perm]) ** 2).sum(-1).mean()) <= tol)


# --------------------------------------------------------------------------------------
# The target
# --------------------------------------------------------------------------------------
def fragment_targets(
    mol: Chem.Mol,
    frag: Fragmentation,
    target_pos: np.ndarray,
    template_pos: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    max_atoms: int | None = MAX_LIGAND_ATOMS,
    sym_tol: float = SYMMETRY_TOL,
    tie_tol: float = TIE_TOL,
    limit: int = MAX_RELABELLINGS,
) -> FragmentTargets:
    """Per-fragment SE(3) target: the template of each fragment, placed as well as it can be.

    ``target_pos`` is the true pose, ``template_pos`` the fixed internal geometry (the molecule's
    own conformer by default; ``embed_template`` for the ETKDG geometry inference really has). Each
    fragment is fitted once per relabelling the molecule cannot tell apart, the best fit wins, and
    anything within ``tie_tol`` of it is settled by nearness to ``reference`` -- the identity, i.e.
    the template's own frame, unless one is given. The ties are the relabellings that are rotations
    of the template and the gauge circle of a collinear fragment, which is to say everything the
    fit cannot see. Pass ``reference=R_t`` to resolve against the state of a flow instead.

    Ligands with more than ``max_atoms`` heavy atoms raise instead of returning a target; pass
    ``max_atoms=None`` to compute them anyway.
    """
    n = mol.GetNumAtoms()
    if max_atoms is not None and n > max_atoms:
        raise ValueError(f"{n} heavy atoms > {max_atoms}: out of scope for the fragment parameterisation")
    target = np.asarray(target_pos, dtype=np.float64).reshape(n, 3)
    template = np.asarray(
        mol.GetConformer().GetPositions() if template_pos is None else template_pos, dtype=np.float64
    ).reshape(n, 3)

    num = frag.num_fragments
    ref = np.broadcast_to(np.eye(3) if reference is None else np.asarray(reference, dtype=np.float64), (num, 3, 3))
    classes = symmetry_classes(mol)

    rot = np.zeros((num, 3, 3))
    trans = np.zeros((num, 3))
    center = np.zeros((num, 3))
    pos = np.zeros((n, 3))
    rmsd = np.zeros(num)
    pairing = np.arange(n, dtype=np.int64)
    gauge = np.zeros(num, dtype=bool)
    relabelled = np.zeros(num, dtype=bool)
    num_relabellings = np.zeros(num, dtype=np.int64)
    num_symmetries = np.zeros(num, dtype=np.int64)

    for f in range(num):
        atoms = np.flatnonzero(frag.frag_idx == f)
        centred = template[atoms] - template[atoms].mean(0)
        perms = fragment_relabellings(mol, atoms, classes, limit)

        # One candidate per relabelling, each with its gauge already pinned so that the distance
        # to the reference is the distance to the family rather than to an arbitrary member.
        candidates = []
        for i, perm in enumerate(perms):
            true = target[atoms[perm]]
            r, t = kabsch(centred, true)
            r, free = pin_gauge(r, centred, ref[f])
            fit = float(np.sqrt(((centred @ r.T + t - true) ** 2).sum(-1).mean()))
            candidates.append((fit, rotation_distance(ref[f], r), i, r, t, perm, free))

        best = min(c[0] for c in candidates)
        fit, _, _, r, t, perm, free = min((c for c in candidates if c[0] <= best + tie_tol), key=lambda c: c[1:3])

        rot[f], trans[f], center[f] = r, t, template[atoms].mean(0)
        pos[atoms] = centred @ r.T + t
        rmsd[f] = fit
        pairing[atoms] = atoms[perm]
        gauge[f] = free
        relabelled[f] = not np.array_equal(perm, np.arange(len(atoms)))
        num_relabellings[f] = len(perms)
        num_symmetries[f] = sum(is_template_symmetry(centred, p, sym_tol) for p in perms)

    return FragmentTargets(rot, trans, center, pos, rmsd, pairing, gauge, relabelled, num_relabellings, num_symmetries)


def embed_template(mol: Chem.Mol, seed: int = 0xC0FFEE) -> np.ndarray:
    """A fresh ETKDG conformer's heavy-atom coordinates, in the molecule's own atom order.

    The internal geometry inference actually starts from, as opposed to the crystal pose's. Stereo
    tags are read off the input 3D coordinates first, so the conformer is the same enantiomer and
    the rigid fit is not asked to mirror anything.
    """
    from rdkit.Chem import AllChem

    probe = Chem.Mol(mol)
    if probe.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(probe)
    probe = Chem.AddHs(probe)
    if AllChem.EmbedMolecule(probe, randomSeed=seed) != 0:
        raise ValueError("ETKDG could not embed a template conformer")
    probe = Chem.RemoveHs(probe)
    return probe.GetConformer().GetPositions().astype(np.float64)


def load_template(mol: Chem.Mol, path: str | Path) -> np.ndarray:
    """Read a template conformer of ``mol`` from a file, in ``mol``'s own atom order.

    For the start conformers that ship next to a complex (``*_ligand_start_conf.sdf``), which are
    the geometry inference is handed rather than a freshly embedded one.

    Any explicit hydrogen left over is stripped, which ``Chem.RemoveHs`` does not do for one that
    defines a double bond's stereochemistry -- 1OWE's amidine keeps one, so its start conformer
    reads as a 23-atom molecule against the ligand's 22.
    """
    from pocketdit.chem import load_ligand

    other = Chem.RemoveAllHs(load_ligand(path))
    order = mol.GetSubstructMatch(other)
    if len(order) != mol.GetNumAtoms() or other.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError(f"{path} is not the same molecule as the ligand ({other.GetNumAtoms()} atoms)")
    pos = np.empty((mol.GetNumAtoms(), 3))
    pos[list(order)] = other.GetConformer().GetPositions()
    return pos


FRAGMENT_KINDS = ("aromatic ring", "saturated ring", "acyclic")


def fragment_kinds(mol: Chem.Mol, frag: Fragmentation) -> list[str]:
    """What kind of thing each fragment is, which is what the fit residual tracks.

    A saturated ring is rigid in the parameterisation but not in reality: its pucker is a
    conformational degree of freedom the template fixes and the fit cannot recover.
    """
    out = []
    for f in range(frag.num_fragments):
        atoms = np.flatnonzero(frag.frag_idx == f)
        ring = [mol.GetAtomWithIdx(int(a)) for a in atoms]
        if not any(a.IsInRing() for a in ring):
            out.append("acyclic")
        else:
            out.append("aromatic ring" if any(a.GetIsAromatic() for a in ring) else "saturated ring")
    return out


def joint_errors(mol: Chem.Mol, frag: Fragmentation, pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """How much each cut bond's length is off in the target pose (Å).

    The one error the per-fragment fits do not control. Fragments are fitted independently, so
    nothing holds the two halves of a cut bond at bonding distance; this is how far the assembled
    target drifts from the true geometry at its joints.
    """
    pos, target = np.asarray(pos, dtype=np.float64), np.asarray(target_pos, dtype=np.float64)
    out = []
    for bond_idx in frag.cut_bonds:
        bond = mol.GetBondWithIdx(int(bond_idx))
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        out.append(abs(np.linalg.norm(pos[u] - pos[v]) - np.linalg.norm(target[u] - target[v])))
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Statistics over a dataset
# --------------------------------------------------------------------------------------
def statistics(
    paths: list[Path],
    selection: str = "compact",
    template: str = "etkdg",
    max_atoms: int | None = MAX_LIGAND_ATOMS,
) -> dict:
    """Fragment-target statistics over a list of ligand files, printed by ``report``.

    ``template="etkdg"`` fits a fresh conformer, which is the honest measurement: it is the
    geometry inference has. ``template="crystal"`` fits the true pose onto itself, where every
    residual has to be zero -- a check on the fit rather than a measurement of anything. Anything
    else is taken as a glob next to each ligand, e.g. ``"*_ligand_start_conf.sdf"``.
    """
    from pocketdit.chem import load_ligand

    rows, failures, filtered = [], [], 0
    for path in paths:
        try:
            mol = load_ligand(path)
            if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
                filtered += 1
                continue
            true = mol.GetConformer().GetPositions().astype(np.float64)
            frag = fragment(mol, selection=selection)
            if template == "crystal":
                start = true
            elif template == "etkdg":
                start = embed_template(mol)
            else:
                start = load_template(mol, next(iter(sorted(path.parent.glob(template)))))
            tgt = fragment_targets(mol, frag, true, start, max_atoms=max_atoms)
        except Exception as exc:  # noqa: BLE001 - a corrupt input should not stop the sweep
            failures.append((path, str(exc)))
            continue
        rows.append(
            {
                "name": path.parent.name or path.stem,
                "atoms": mol.GetNumAtoms(),
                "fragments": frag.num_fragments,
                "sizes": frag.sizes,
                "rmsd": tgt.rmsd,
                "kinds": fragment_kinds(mol, frag),
                "total_rmsd": tgt.total_rmsd(true),
                "joints": joint_errors(mol, frag, tgt.pos, true),
                "gauge": int(tgt.gauge.sum()),
                "relabelled": int(tgt.relabelled.sum()),
                "symmetric": int((tgt.num_symmetries > 1).sum()),
                "relabellable": int((tgt.num_relabellings > 1).sum()),
            }
        )
    return {
        "rows": rows,
        "failures": failures,
        "filtered": filtered,
        "selection": selection,
        "template": template,
        "max_atoms": max_atoms,
    }


def _percentiles(values: list[float] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "n/a"
    q = np.percentile(arr, [50, 90, 99])
    return f"mean {arr.mean():6.3f}  median {q[0]:6.3f}  p90 {q[1]:6.3f}  p99 {q[2]:6.3f}  max {arr.max():6.3f}"


def report(stats: dict) -> None:
    rows = stats["rows"]
    if not rows:
        print("no molecules parsed")
        for path, exc in stats["failures"][:10]:
            print(f"  {path}: {exc}")
        return

    n_frags = sum(r["fragments"] for r in rows)
    fit = np.concatenate([r["rmsd"] for r in rows])
    joints = np.concatenate([r["joints"] for r in rows]) if any(len(r["joints"]) for r in rows) else np.zeros(0)
    print(
        f"molecules          {len(rows)}  (over {stats['max_atoms']} heavy atoms: {stats['filtered']}, "
        f"failed to parse: {len(stats['failures'])})"
    )
    print(f"selection          {stats['selection']}   template {stats['template']}")
    print(f"fragments          {n_frags}")
    print()
    print(f"fragment fit RMSD  {_percentiles(fit)}  Å")
    kinds = [k for r in rows for k in r["kinds"]]
    for kind in FRAGMENT_KINDS:
        sel = fit[[k == kind for k in kinds]]
        print(f"  {kind:15s}{_percentiles(sel)}  Å  ({len(sel)} fragments)")
    print(f"pose RMSD          {_percentiles([r['total_rmsd'] for r in rows])}  Å   <- the fragment-mode floor")
    print(f"cut bond length    {_percentiles(joints)}  Å")
    print()
    gauge = sum(r["gauge"] for r in rows)
    symmetric = sum(r["symmetric"] for r in rows)
    relabellable = sum(r["relabellable"] for r in rows)
    relabelled = sum(r["relabelled"] for r in rows)
    print(f"gauge pinned       {gauge:6d}  {100 * gauge / n_frags:5.1f}%  of fragments, collinear or single-atom")
    print(f"relabellable       {relabellable:6d}  {100 * relabellable / n_frags:5.1f}%  have any relabelling at all")
    print(
        f"  rotations of it  {symmetric:6d}  {100 * symmetric / n_frags:5.1f}%  which tie exactly, settled by nearness"
    )
    print(f"relabelled         {relabelled:6d}  {100 * relabelled / n_frags:5.1f}%  took a non-identity relabelling")
    if stats["failures"]:
        print()
        print("parse failures")
        for path, exc in stats["failures"][:10]:
            print(f"  {path}: {exc}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from pocketdit.fragment import SELECTIONS

    p = argparse.ArgumentParser(description="Per-fragment SE(3) target statistics over a ligand dataset")
    p.add_argument("--root", type=Path, help="dataset root, one subfolder per complex")
    p.add_argument("--glob", default="*/*_ligand.sdf", help="ligand glob relative to --root")
    p.add_argument("--smiles", nargs="*", help="look at these SMILES instead of a dataset")
    p.add_argument("--selection", default="compact", choices=SELECTIONS)
    p.add_argument(
        "--template",
        default="etkdg",
        help='internal geometry to place: "etkdg" for a fresh conformer, "crystal" for the true '
        'pose itself, or a glob next to each ligand such as "*_ligand_start_conf.sdf"',
    )
    p.add_argument("--max_atoms", type=int, default=MAX_LIGAND_ATOMS, help="skip ligands with more heavy atoms")
    p.add_argument("--limit", type=int, help="only look at the first N ligands")
    args = p.parse_args(argv)

    if args.smiles:
        for smiles in args.smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                p.error(f"could not parse {smiles!r}")
            mol = Chem.RemoveHs(Chem.Mol(mol))
            true = embed_template(mol, seed=1)
            frag = fragment(mol, selection=args.selection)
            start = true if args.template == "crystal" else embed_template(mol)
            tgt = fragment_targets(mol, frag, true, start, max_atoms=args.max_atoms)
            print(f"{smiles}")
            print(
                f"  {frag.num_fragments} fragments, sizes {frag.sizes.tolist()}, pose RMSD {tgt.total_rmsd(true):.3f} Å"
            )
            for f in range(frag.num_fragments):
                flags = "".join(c for c, on in (("g", tgt.gauge[f]), ("r", tgt.relabelled[f])) if on)
                print(
                    f"    frag {f}  {int((frag.frag_idx == f).sum()):2d} atoms  fit {tgt.rmsd[f]:6.3f} Å  "
                    f"turn {math.degrees(rotation_angle(tgt.rot[f])):5.1f}°  "
                    f"relabellings {tgt.num_relabellings[f]:2d} "
                    f"({tgt.num_symmetries[f]} rotations)  {flags}"
                )
        return

    if not args.root:
        p.error("pass either --root or --smiles")
    paths = sorted(args.root.glob(args.glob))[: args.limit]
    if not paths:
        p.error(f"no ligands matched {args.root / args.glob}")
    report(statistics(paths, args.selection, args.template, args.max_atoms))


if __name__ == "__main__":
    main()
