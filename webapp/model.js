/* CatBoost inference + exact path-dependent TreeSHAP for oblivious trees (raw log-scale output for Poisson models).
   Mirrors src/export_web_model.py (verified against catboost.predict and shap.TreeExplainer). */
(function (root) {
  function create(M) {
    const F = M.features.length;
    const fact = [1, 1, 2, 6, 24, 120];
    // Pre-process trees: distinct features and Shapley weights
    const trees = M.trees.map(t => {
      const U = [...new Set(t.f)];
      return { f: t.f, b: t.b, v: t.v, w: t.w, U, depth: t.f.length };
    });

    function leafIndex(t, x) {
      let idx = 0;
      for (let d = 0; d < t.depth; d++) if (x[t.f[d]] > t.b[d]) idx |= 1 << d;
      return idx;
    }

    function predict(x) {
      let s = M.bias;
      for (const t of trees) s += t.v[leafIndex(t, x)];
      return s;
    }

    function cover(t, mask, bits) {
      let c = 0;
      for (let l = 0; l < t.w.length; l++) if ((l & mask) === bits) c += t.w[l];
      return c;
    }

    // E[f | x_S] with unknown features split by training cover; root = last split (CatBoost order)
    function condExp(t, x, inS) {
      function rec(k, mask, bits, p) {
        if (k === t.depth) return p * t.v[bits];
        const d = t.depth - 1 - k, bit = 1 << d, f = t.f[d];
        if (inS(f)) return rec(k + 1, mask | bit, bits | (x[f] > t.b[d] ? bit : 0), p);
        const tot = cover(t, mask, bits);
        if (tot === 0) return 0;
        let out = 0;
        for (const c of [0, bit]) {
          const cw = cover(t, mask | bit, bits | c);
          if (cw) out += rec(k + 1, mask | bit, bits | c, p * cw / tot);
        }
        return out;
      }
      return rec(0, 0, 0, 1);
    }

    function shap(x) {
      const phi = new Array(F).fill(0);
      let base = M.bias;
      for (const t of trees) {
        const U = t.U, k = U.length, n = 1 << k;
        const v = new Array(n);
        for (let s = 0; s < n; s++) v[s] = condExp(t, x, f => { const i = U.indexOf(f); return i >= 0 && (s >> i) & 1; });
        base += v[0];
        for (let i = 0; i < k; i++) {
          const bi = 1 << i;
          for (let s = 0; s < n; s++) {
            if (s & bi) continue;
            let size = 0; for (let j = 0; j < k; j++) if ((s >> j) & 1) size++;
            phi[U[i]] += (fact[size] * fact[k - size - 1] / fact[k]) * (v[s | bi] - v[s]);
          }
        }
      }
      return { base, phi };
    }

    return { predict, shap, features: M.features };
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { create };
  else root.TreeModel = { create };
})(this);
