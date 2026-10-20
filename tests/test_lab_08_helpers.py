"""Independent, always-on tests for lab 08 (CI-safe: no download, no model).

`tests/test_lab_08_scaffold.py` executes the real student notebook and is
marked `network` because its very first code cell (section 1,
`tvd.EuroSAT(..., download=True)`) runs *before* the first student TODO --
unlike modules 09/10, there is no way to reach even the earliest TODO
without a network call. Scheduled network CI runs that notebook test; this
file protects the fast PR suite without reimplementing the lab's own
deliverables.

Two tests below (`test_toy_retrieval_fixture_*` and
`test_gallery_query_split_logic_*`) read the *actual, current* source of
specific cells out of `labs/08_embeddings/lab.ipynb` via `nbformat` and
execute it (the second with a tiny synthetic dataset substituted for
EuroSAT, per the "controlled data-source substitution" the lab brief
allows). Both cells are "gotowe" (provided, not a student TODO), so
running them reveals nothing about the student's tasks -- but because
they read the live notebook rather than a hardcoded copy, a future edit to
either cell that breaks something is exactly what makes these tests fail.
That is the main thing distinguishing them from the four formula-sanity
tests above them, which encode already-public lecture formulas
(epsilon-safe L2 normalization, cosine similarity) checked against
PyTorch's own reference or basic algebra -- useful as living
documentation and as a check on this test file's own arithmetic, but they
do **not** read any lab/lecture file and so cannot catch a regression
there. Said plainly: if this concerns you, the formula-sanity tests are
the weaker half of this file; the two notebook-reading tests are the
stronger half.

What is **not** tested here, and why: `l2_normalize`, `search`, and
`precision_at_k` are the lab's actual deliverables -- a general,
reusable version of any of them, even written independently rather than
copied from `decisions/solutions/08_embeddings/`, would still amount to
publishing the solution. Ranking under ties, excluding a query that
happens to coincide with a gallery vector, varying k end-to-end, and
Precision@k computed from a real ranking are therefore validated only
**locally**, by running the private reference solution
(`decisions/solutions/08_embeddings/lab_solution.ipynb`) against small
synthetic cases with a known answer -- not in this file, and not in any
file committed to the repository. See the PR description for that run's
results.
"""

from pathlib import Path

import nbformat
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_NOTEBOOK_PATH = REPO_ROOT / "labs" / "08_embeddings" / "lab.ipynb"


def _read_cell_source(notebook_path, contains):
    nb = nbformat.read(notebook_path, as_version=4)
    matches = [c.source for c in nb.cells if c.cell_type == "code" and contains in c.source]
    assert len(matches) == 1, (
        f"expected exactly one cell containing {contains!r}, found {len(matches)}"
    )
    return matches[0]


def test_l2_normalize_epsilon_pattern_handles_zero_row_without_nan():
    """Generalizes the lecture's already-public single-vector formula
    (`e / (e.norm() + eps)`, lectures/08_embeddings/lecture.md) to a batch
    of rows -- a mechanical extension, not the lab's own `l2_normalize`
    (which uses `clamp_min`, a different safe-division strategy)."""
    X = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    eps = 1e-8
    normalized = X / (X.norm(dim=1, keepdim=True) + eps)

    assert not torch.isnan(normalized).any()
    assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(normalized[1], torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized[2].norm(), torch.tensor(1.0), atol=1e-6)


def test_l2_normalize_epsilon_pattern_preserves_unit_norm_for_small_vectors():
    """A vector with a small (but nonzero) norm, large relative to eps,
    should still end up at unit length, not blow up or underflow to zero."""
    small = torch.tensor([[1e-3, 0.0]])
    normalized = small / (small.norm(dim=1, keepdim=True) + 1e-8)
    assert torch.allclose(normalized.norm(dim=1), torch.tensor([1.0]), atol=1e-4)


def test_l2_normalize_epsilon_pattern_loses_precision_when_norm_approaches_eps():
    """When the vector's own norm is only ~100x the epsilon, the epsilon
    is no longer negligible and the result is measurably short of unit
    norm -- a real, bounded trade-off of the "add eps" safety pattern
    (not a bug), worth knowing before picking eps relative to expected
    embedding scales."""
    tiny = torch.tensor([[1e-6, 0.0]])
    eps = 1e-8
    normalized = tiny / (tiny.norm(dim=1, keepdim=True) + eps)
    resulting_norm = normalized.norm(dim=1).item()
    assert not torch.isnan(normalized).any()
    assert 0.9 < resulting_norm < 1.0, (
        f"expected a measurable (but bounded) shortfall from unit norm, got {resulting_norm}"
    )


def test_cosine_similarity_formula_matches_pytorch_builtin():
    """The cosine-similarity formula from the lecture (a.b / (|a||b|))
    reimplemented from scratch must agree with PyTorch's own
    F.cosine_similarity -- a check against an independent reference, not
    against the lab's search()."""
    torch.manual_seed(0)
    a = torch.randn(5, 8)
    b = torch.randn(5, 8)

    manual = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1))
    reference = F.cosine_similarity(a, b, dim=1)
    assert torch.allclose(manual, reference, atol=1e-6)


def test_toy_retrieval_fixture_in_notebook_has_unambiguous_ranking():
    """The hand-constructed example in labs/08_embeddings/lab.ipynb (6
    vectors, 2 classes) is the known-answer check students run their own
    search()/precision_at_k() against. This reads that cell's *current*
    source directly out of the notebook (not a hardcoded copy) and
    confirms the fixture is genuinely unambiguous -- no ties at the k=1
    or k=3 boundary -- and that its own already-published `expected`
    top-3 ranking is reproduced by plain cosine similarity, computed here
    once for this one fixed case, not via a reusable ranking function."""
    source = _read_cell_source(LAB_NOTEBOOK_PATH, "toy_gallery = torch.tensor")
    data_source = source.split("toy_result")[0]  # stop before the call to the student's search()

    namespace = {"torch": torch}
    exec(data_source, namespace)
    toy_gallery = namespace["toy_gallery"]
    toy_gallery_y = namespace["toy_gallery_y"]
    toy_query = namespace["toy_query"]

    g = toy_gallery / toy_gallery.norm(dim=1, keepdim=True)
    q = toy_query / toy_query.norm()
    sims = g @ q

    sorted_sims, order = sims.sort(descending=True)
    assert (sorted_sims[:-1] - sorted_sims[1:] > 1e-4).all(), "toy fixture has a tie"

    assert order[:3].tolist() == [0, 1, 2], (
        "top-3 must match the notebook's own published `expected = [0, 1, 2]`"
    )
    assert toy_gallery_y[order[:3]].eq(0).all(), "top-3 should all be class A -> Precision@3 == 1.0"


def test_gallery_query_split_logic_in_notebook_is_disjoint_with_synthetic_dataset():
    """The gallery/query split (labs/08_embeddings/lab.ipynb, section 1) is
    provided code, not a student TODO -- the student's only task there is
    to verify disjointness, not to build the split. This reads that
    cell's actual, current source and executes it with a tiny synthetic
    dataset substituted for `tvd.EuroSAT(...)` (no download), confirming
    the real splitting logic -- not a reimplementation of it -- produces
    disjoint, correctly-sized gallery/query index sets."""
    source = _read_cell_source(LAB_NOTEBOOK_PATH, "GALLERY_PER_CLASS")
    # cut the `with contextlib.redirect_stdout(...): full_ds = tvd.EuroSAT(...)` block;
    # everything from "CLASSES = full_ds.classes" onward only needs `full_ds` to exist
    split_logic = "CLASSES" + source.split("CLASSES", 1)[1]

    # the notebook's split loop hardcodes range(10) (EuroSAT has 10 classes), not
    # len(full_ds.classes) -- the fake dataset must match that, not an arbitrary count
    n_classes, per_class = 10, 50
    fake_ds = type(
        "FakeDataset",
        (),
        {
            "classes": list(range(n_classes)),
            "__iter__": lambda self: iter(
                (None, c) for c in range(n_classes) for _ in range(per_class)
            ),
        },
    )()

    namespace = {"full_ds": fake_ds}
    exec(split_logic, namespace)

    gallery_indices = namespace["gallery_indices"]
    query_indices = namespace["query_indices"]
    gallery_per_class = namespace["GALLERY_PER_CLASS"]
    query_per_class = namespace["QUERY_PER_CLASS"]

    assert len(set(gallery_indices) & set(query_indices)) == 0
    assert len(gallery_indices) == n_classes * gallery_per_class
    assert len(query_indices) == n_classes * query_per_class
