"""Static guard against the failure that has bitten this project five times.

A string-replacement patch that doesn't match fails silently. When the *use* of
a variable lands but its *definition* doesn't, the file still compiles and still
imports. It only dies at runtime, often weeks later when a branch first executes.

This walks each function and reports any local name that is read before any
assignment to it, accounting for enclosing scopes, comprehensions and globals.
"""
import ast, builtins, sys

def module_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            out |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split('.')[0] for a in n.names}
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    out.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    out |= {a.asname or a.name.split('.')[0] for a in sub.names}
    return out

def bound(fn):
    """Every name that gets bound anywhere inside fn, including nested defs."""
    names = set()
    for a in list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs):
        names.add(a.arg)
    if fn.args.vararg: names.add(fn.args.vararg.arg)
    if fn.args.kwarg:  names.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            if n is not fn:
                for a in getattr(n.args, 'args', []): names.add(a.arg)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {a.asname or a.name.split('.')[0] for a in n.names}
        elif isinstance(n, (ast.comprehension,)):
            names |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.Global):
            names |= set(n.names)
        elif isinstance(n, ast.Nonlocal):
            names |= set(n.names)
    return names

bad = 0
for path in sys.argv[1:]:
    tree = ast.parse(open(path).read())
    mod = module_names(tree)
    # map each function to its chain of enclosing functions
    parents = {}
    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parents[child] = chain
                walk(child, chain + [child])
            else:
                walk(child, chain)
    walk(tree, [])
    issues = []
    for fn, chain in parents.items():
        avail = set(mod)
        for enc in chain:
            avail |= bound(enc)
        avail |= bound(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in avail and not hasattr(builtins, n.id):
                    issues.append((n.lineno, fn.name, n.id))
    issues = sorted(set(issues))
    print(f"{path}: {len(issues)} name(s) used but never bound")
    for line, fname, name in issues:
        print(f"  line {line} in {fname}(): '{name}'")
    bad += len(issues)

print()
print("PASS: every name has a definition" if not bad else f"FAIL: {bad} undefined")
sys.exit(1 if bad else 0)
