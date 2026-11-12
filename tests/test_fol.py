from reasoner.fol import parse_fol, three_way, to_clauses, refute


def P(xs):
    return [parse_fol(x) for x in xs]


def test_assignment_example():
    kb = P(["Young(bob)", "Kind(bob)", "∀x (Young(x) → Smart(x))",
            "∀x (Smart(x) ∧ Kind(x) → Successful(x))"])
    assert three_way(kb, parse_fol("Successful(bob)")) == "True"
    assert three_way(kb, parse_fol("¬Successful(bob)")) == "False"
    assert three_way(kb, parse_fol("Tall(bob)")) == "Uncertain"


def test_existential_and_xor():
    kb = P(["∀x (Dog(x) → Animal(x))", "∃x (Dog(x) ∧ Brown(x))", "∀x (Animal(x) ⊕ Plant(x))"])
    assert three_way(kb, parse_fol("∃x (Animal(x) ∧ Brown(x))")) == "True"
    assert three_way(kb, parse_fol("∃x (Plant(x) ∧ Dog(x))")) == "False"
    assert three_way(kb, parse_fol("∃x (Plant(x) ∧ Brown(x))")) == "Uncertain"


def test_transitivity_multi_quantifier():
    kb = P(["∀x ∀y (Parent(x, y) → Ancestor(x, y))",
            "∀x ∀y ∀z (Ancestor(x, y) ∧ Ancestor(y, z) → Ancestor(x, z))",
            "Parent(a, b)", "Parent(b, c)", "Parent(c, d)"])
    assert three_way(kb, parse_fol("Ancestor(a, d)")) == "True"
    assert three_way(kb, parse_fol("Ancestor(d, a)")) == "Uncertain"


def test_skolem_functions():
    kb = P(["∀x ∃y Loves(x, y)", "∀x ∀y (Loves(x, y) → Happy(x))"])
    assert three_way(kb, parse_fol("∀x Happy(x)")) == "True"


def test_iff_and_ascii_syntax():
    kb = P(["all x (Bird(x) <-> (Feathers(x) & -Mammal(x)))", "Feathers(tweety)", "-Mammal(tweety)"])
    assert three_way(kb, parse_fol("Bird(tweety)")) == "True"


def test_proof_trace():
    ax = [c for f in P(["Young(bob)", "∀x (Young(x) → Smart(x))"]) for c in to_clauses(f)]
    tr = []
    assert refute(ax, to_clauses(parse_fol("¬Smart(bob)")), trace=tr) is True
    assert tr and not tr[-1][2]  # last step derives the empty clause


def test_long_xor_chain_is_fast():
    import time
    names = [f"C{i}" for i in range(14)]
    chain = " ⊕ ".join(f"{n}(x)" for n in names)
    # a left-associated ⊕ chain is parity: all but the last false forces the last true
    kb = P([f"∀x (College(x) → ({chain}))", "College(a)"] + [f"¬{n}(a)" for n in names[:-1]])
    t = time.perf_counter()
    assert three_way(kb, parse_fol(f"{names[-1]}(a)")) == "True"
    assert three_way(P([f"∀x (College(x) → ({chain}))", "College(a)", "C3(a)"]), parse_fol("¬C5(a)")) == "Uncertain"
    assert time.perf_counter() - t < 5


def test_equality_case_split():
    kb = P(["∀x (Man(x) ∧ ¬(x=michael) → Taller(michael, x))",
            "∀x ∀y ∀z ((Taller(x, y) ∧ Taller(y, z)) → Taller(x, z))",
            "Man(peter) ∧ Taller(peter, michael)"])
    assert three_way(kb, parse_fol("∀x (Man(x) → Taller(peter, x) ∨ x=peter)")) == "True"
