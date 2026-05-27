import re, os, sys
t = open(sys.argv[1], encoding="utf-8").read()
cg = re.findall(r"\\cite\{([^}]*)\}", t)
cites = {c for g in cg for c in g.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
labels = set(re.findall(r"\\label\{([^}]*)\}", t))
refs = set(re.findall(r"\\ref\{([^}]*)\}", t))
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", t)
print("cites not in bib :", (cites - bibs) or "OK")
print("refs not labelled:", (refs - labels) or "OK")
print("labels unused    :", (labels - refs) or "OK")
print("braces balance   :", t.count("{") - t.count("}"))
base = os.path.dirname(os.path.abspath(sys.argv[1]))
for f in figs:
    ok = any(os.path.exists(os.path.join(base, p, f)) for p in ("../figures", "./figures"))
    print(("fig OK : " if ok else "FIG MISSING: ") + f)
