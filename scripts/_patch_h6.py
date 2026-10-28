"""One-off: sync the presentation and appendix to the four-phase cycle with H6 quarterly + 3M exit."""
import pathlib

root = pathlib.Path(__file__).resolve().parent


def patch(path, pairs, insertions=None):
    p = root / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, path + " :: " + a[:70]
        s = s.replace(a, b)
    for anchor, add in (insertions or []):
        assert anchor in s, path + " :: " + anchor[:70]
        s = s.replace(anchor, anchor + add, 1)
    p.write_text(s, encoding="utf-8")
    print("patched", path)


# ---------------- README ----------------
patch("build_readme.py", [
    ('{figure_block("Restructuring Decision Flow - How RQ01, RQ02 and RQ03 Control Allocation"',
     '{figure_block("Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom"'),
    ('"Layer A carries the financial state: P_N and P_S are complementary by construction, equal to the "\n'
     '              "weekly entry hazard in normal states and the weekly exit hazard in stress states. Layer B carries the "\n'
     '              "transition clock: entry, the last-short flag, the lock, the bottom gates and exit. Layer C carries the "\n'
     '              "risk and weight constraints. The engine applies them in order, so a favourable signal produces no "\n'
     '              "trade while the lock is active or when the risk caps make the target infeasible.",',
     '"RQ01 identifies the state, RQ02 supplies Pn and Ps from the H6 quarterly pre-onset model and the 3-month "\n'
     '              "exit model, and RQ03 applies the risk cap, volatility and z-score constraints. The portfolio is "\n'
     '              "locked between last short and bottom confirmation, so a favourable signal produces no trade "\n'
     '              "while the lock is active.",'),
    ('''| Restructuring state | RQ01 inputs | RQ02 inputs | RQ03 conditions | Permitted action |
|---|---|---|---|---|
| Normal | pooled rule calm; P_S = weekly entry hazard | entry model applies outside phases | S1 band 80-100% equity; ES cap 2% | Common restructuring if feasible |
| Stress, pre-lock | pooled rule fires | short system votes | S2 bond band 50-80%, cash residual | Common restructuring until the lock |
| Short confirmed | state persists | last-short flag valid, max 18 months | lock active; band allocator withdrawn | Restructuring prohibited (maintain) |
| Awaiting bottom | state persists | bottom gates unmet | lock maintained; pre-bottom supplement allowed | Maintain the lock |
| Bottom confirmed | state persists | F1-F3: two of three agree; all four gates pass | equity 100%; ES cap bypassed in S3 | Bottom-specific restructuring |
| Exit / post-exit | state updated | p_exit >= 0.027174 | S4 band 50-100% equity, bonds 0-25% | Common mechanism resumes |''',
     '''| Phase | RQ01 | RQ02 | RQ03 | Allocation |
|---|---|---|---|---|
| Normal | pooled rule calm | P_S = weekly hazard of the H6 quarterly pre-onset probability | equity band 80-100% | equity 80-100%, bonds 0-20%, cash residual |
| Last short | state persists | five-vote short system fires; P_N = 1 - P_S | lock: band allocator withdrawn | bonds 50-80%, no equity, restructuring prohibited |
| Bottom | state persists | gates P_N >= 0.995, P_S <= 0.02, z >= 2, drawdown <= 5%; F1-F3 two of three | ES cap bypassed in S3 | equity 100%, profit ladder active |
| Post-bottom | state updated | exit 3M p >= 0.027174 or ladder complete | S4 band | equity 50-100%, bonds 0-25%; 26 normal weeks re-arm the cycle |'''),
])

# ---------------- site ----------------
patch("build_pages.py", [
    ('alt="Restructuring Decision Flow - How RQ01, RQ02 and RQ03 Control Allocation">',
     'alt="Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom">'),
    ('Restructuring Decision Flow - How RQ01, RQ02 and RQ03 Control Allocation',
     'Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom'),
    ('"Layer A carries the financial state (P_N and P_S are complementary by construction); Layer B carries the "\n'
     '    "transition clock and the lock; Layer C carries the risk and weight constraints. The engine applies them in "\n'
     '    "order, so a favourable signal produces no trade while the lock is active or when the risk caps make the target "\n'
     '    "infeasible.",',
     '"RQ01 identifies the state, RQ02 supplies Pn and Ps from the H6 quarterly pre-onset model and the 3-month exit "\n'
     '    "model, and RQ03 applies the risk cap, volatility and z-score constraints. The portfolio is locked between "\n'
     '    "last short and bottom confirmation, so a favourable signal produces no trade while the lock is active.",'),
    ('''<thead><tr><th>Restructuring state</th><th>RQ01 inputs</th>'
    '<th>RQ02 inputs</th><th>RQ03 conditions</th><th>Permitted action</th></tr></thead><tbody>'
    '<tr><td>Normal</td><td>pooled rule calm; P_S = weekly entry hazard</td><td>entry model applies outside phases</td>'
    '<td>S1 band 80-100% equity; ES cap 2%</td><td>Common restructuring if feasible</td></tr>'
    '<tr><td>Stress, pre-lock</td><td>pooled rule fires</td><td>short system votes</td>'
    '<td>S2 bond band 50-80%, cash residual</td><td>Common restructuring until the lock</td></tr>'
    '<tr><td>Short confirmed</td><td>state persists</td><td>last-short flag valid, max 18 months</td>'
    '<td>lock active; band allocator withdrawn</td><td>Restructuring prohibited (maintain)</td></tr>'
    '<tr><td>Awaiting bottom</td><td>state persists</td><td>bottom gates unmet</td>'
    '<td>lock maintained; pre-bottom supplement allowed</td><td>Maintain the lock</td></tr>'
    '<tr><td>Bottom confirmed</td><td>state persists</td><td>F1-F3: two of three agree; all four gates pass</td>'
    '<td>equity 100%; ES cap bypassed in S3</td><td>Bottom-specific restructuring</td></tr>'
    '<tr><td>Exit / post-exit</td><td>state updated</td><td>p_exit >= 0.027174</td>'
    '<td>S4 band 50-100% equity, bonds 0-25%</td><td>Common mechanism resumes</td></tr>'
    '</tbody></table></div>''',
     '''<thead><tr><th>Phase</th><th>RQ01</th><th>RQ02</th><th>RQ03</th><th>Allocation</th></tr></thead><tbody>'
    '<tr><td>Normal</td><td>pooled rule calm</td><td>P_S = weekly hazard of the H6 quarterly pre-onset probability</td>'
    '<td>equity band 80-100%</td><td>equity 80-100%, bonds 0-20%, cash residual</td></tr>'
    '<tr><td>Last short</td><td>state persists</td><td>five-vote short system fires; P_N = 1 - P_S</td>'
    '<td>lock: band allocator withdrawn</td><td>bonds 50-80%, no equity, restructuring prohibited</td></tr>'
    '<tr><td>Bottom</td><td>state persists</td><td>gates P_N &gt;= 0.995, P_S &lt;= 0.02, z &gt;= 2, drawdown &lt;= 5%; F1-F3 two of three</td>'
    '<td>ES cap bypassed in S3</td><td>equity 100%, profit ladder active</td></tr>'
    '<tr><td>Post-bottom</td><td>state updated</td><td>exit 3M p &gt;= 0.027174 or ladder complete</td>'
    '<td>S4 band</td><td>equity 50-100%, bonds 0-25%; 26 normal weeks re-arm the cycle</td></tr>'
    '</tbody></table></div>'''),
])

# ---------------- appendix: A.5 rows use H6 quarterly + 3M ----------------
patch("appendix_data.py", [
    ('_row("RQ01", "A.5", "Normal state definition", "pooled decision calm; P_N = 1 - P_S", "probability",',
     '_row("RQ01", "A.5", "Normal state definition", "pooled decision calm; P_N = 1 - P_S", "probability",'),
    ('"When the pooled rule is calm the overlay sets P_S to the weekly entry hazard 1-(1-p12)^(1/52), so P_N is the "\n'
     '     "probability that stress does not start within the week", "RAW_CODE/rq03_standard/overlay_rq02_exit3_approved.py:80"),',
     '"When the pooled rule is calm, P_S is the weekly hazard of the H6 quarterly pre-onset probability "\n'
     '     "1-(1-p_H6)^(1/78), so P_N is the probability that stress does not start within the week. The published "\n'
     '     "pre-onset choice is H6 quarterly (18-month model); the shipped overlay script reads its H12 column, which "\n'
     '     "is disclosed in the audit (D15).", "results/rq03_parameters.csv; THRESHOLD/RQ02/rq02_selected.json"),'),
])

# ---------------- new discrepancy D15 ----------------
p = root / "appendix_data.py"
s = p.read_text(encoding="utf-8")
anchor = '    {"id": "D14"'
assert anchor in s
d15 = '''    {"id": "D15", "severity": "medium", "area": "RQ02/RQ03",
     "finding": "The stated pre-onset choice is the H6 quarterly model (18 months), but the shipped RQ03 overlay "
                "script reads its H12 column when rebuilding Pn/Ps. The public presentation uses H6 quarterly per "
                "the owner's choice; the script line is recorded here so the two are never silently merged.",
     "evidence": "RAW_CODE/rq03_standard/overlay_rq02_exit3_approved.py (H12_COL usage)",
     "action": "Align the overlay column with the stated choice or document the divergence in the next release."},
'''
s = s.replace(anchor, d15 + anchor, 1)
p.write_text(s, encoding="utf-8")
print("D15 added")
