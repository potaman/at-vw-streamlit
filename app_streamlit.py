#!/usr/bin/env python
# coding: utf-8
import io, os, zipfile, json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from skimage import color, util, exposure, restoration, measure, morphology

# ===================== Page / Config =====================
st.set_page_config(page_title="AT + VW Simplification", layout="wide")

# ===================== Shims (compat layer) =====================
try:
    cache_data = st.cache_data
except AttributeError:
    cache_data = st.cache

def cache_data_deco(**kwargs):
    try:
        return cache_data(**kwargs)
    except TypeError:
        # Older st.cache doesn't accept show_spinner, etc.
        return cache_data()

def show_image(img, **kwargs):
    """Streamlit image with backwards compatibility."""
    try:
        st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        st.image(img, use_column_width=True, **kwargs)

def show_pyplot(fig):
    """Streamlit pyplot with backwards compatibility."""
    try:
        st.pyplot(fig, clear_figure=True)
    except TypeError:
        st.pyplot(fig)

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >=10
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS             # Older Pillow

def caption(txt):
    try:
        st.caption(txt)
    except AttributeError:
        st.write(txt)

# ===================== Utilities =====================
def to_gray(img):
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = color.rgb2gray(arr)
    return util.img_as_float(arr)

def clahe(img, clip_limit=0.01, nbins=256):
    return exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=nbins)

def grayscale_levels_cmap(levels):
    colors = [ (1 - i/(levels-1),) * 3 for i in range(levels) ] if levels>1 else [(1,1,1)]
    return ListedColormap(colors, name=f'grays_{levels}_white0')

def _level_colors(levels):
    base = plt.cm.get_cmap('tab10')
    cols = base(np.linspace(0, 1, min(levels, 10)))
    if levels > 10:
        reps = int(np.ceil(levels / 10))
        cols = np.vstack([cols for _ in range(reps)])[:levels]
    return cols

# --- RDP (fallback) ---
def rdp(points, epsilon):
    points = np.asarray(points)
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    lsq = np.dot(line, line)
    if lsq == 0:
        dists = np.linalg.norm(points - start, axis=1)
        idx = np.argmax(dists)
        return points[[0, idx, -1]] if dists[idx] > 0 else points[[0, -1]]
    def dist_sq(p):
        v = p - start
        cross = np.abs(np.cross(line, v))
        return (cross**2)/lsq
    d2 = np.array([dist_sq(p) for p in points])
    i = np.argmax(d2)
    if d2[i] > 0 and d2[i] > 1e-12 and d2[i] > epsilon**2:
        left = rdp(points[:i+1], epsilon)
        right = rdp(points[i:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])

# --- Visvalingam–Whyatt simplification ---
def _triangle_area(a, b, c):
    return 0.5 * abs(a[1]*(b[0]-c[0]) + b[1]*(c[0]-a[0]) + c[1]*(a[0]-b[0]))

def visvalingam_whyatt(points, area_thresh=None, keep_ratio=None, preserve_ends=True):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 2: return pts.copy()
    areas = np.full(n, np.inf)
    prev = np.arange(n) - 1
    nxt = np.arange(n) + 1
    prev[0] = -1; nxt[-1] = -1
    for i in range(1, n-1):
        areas[i] = _triangle_area(pts[prev[i]], pts[i], pts[nxt[i]])
    target = None
    if keep_ratio is not None:
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        target = max(2, int(np.ceil(keep_ratio * n)))
    removed = np.zeros(n, dtype=bool)
    def remove(i):
        removed[i] = True
        p = prev[i]; q = nxt[i]
        if p >= 0: nxt[p] = q
        if q >= 0: prev[q] = p
        if p > 0 and q >= 0:
            areas[p] = _triangle_area(pts[prev[p]], pts[p], pts[q])
        if q < n-1 and p >= 0:
            areas[q] = _triangle_area(pts[p], pts[q], pts[nxt[q]])
    while True:
        i = -1; min_area = np.inf; j = nxt[0]
        while j != -1 and j < n-1:
            if not removed[j] and areas[j] < min_area:
                min_area = areas[j]; i = j
            j = nxt[j]
        if i == -1: break
        if area_thresh is not None and min_area >= area_thresh: break
        if target is not None and (n - removed.sum()) <= target: break
        remove(i)
    simplified = pts[~removed]
    if preserve_ends:
        if not np.allclose(simplified[0], pts[0]): simplified = np.vstack([pts[0], simplified])
        if not np.allclose(simplified[-1], pts[-1]): simplified = np.vstack([simplified, pts[-1]])
    return simplified

# ===================== Core pipeline =====================
def ambrosio_tortorelli(f, lam=0.25, alpha=0.06, eps=1.2, n_iters=200, tau_u=0.2, tau_v=0.15):
    f = f.astype(np.float64); u = f.copy(); v = np.ones_like(f)
    def grad_x(a):
        gx = np.zeros_like(a); gx[:, :-1] = a[:,1:] - a[:,:-1]; return gx
    def grad_y(a):
        gy = np.zeros_like(a); gy[:-1, :] = a[1:, :] - a[:-1, :]; return gy
    def div(px, py):
        dx = np.zeros_like(px); dy = np.zeros_like(py)
        dx[:,0] = px[:,0]; dx[:,1:-1] = px[:,1:-1] - px[:,0:-2]; dx[:,-1] = -px[:,-2]
        dy[0,:] = py[0,:]; dy[1:-1,:] = py[1:-1,:] - py[0:-2,:]; dy[-1,:] = -py[-2,:]
        return dx + dy
    for _ in range(n_iters):
        ux = grad_x(u); uy = grad_y(u); v2 = v*v; px = v2 * ux; py = v2 * uy
        smooth = div(px, py); du = (u - f) + lam * smooth; u = np.clip(u - tau_u * du, 0, 1)
        ux = grad_x(u); uy = grad_y(u); lap_v = div(grad_x(v), grad_y(v))
        dv = 2.0*v*(ux*ux + uy*uy) + alpha*( -2.0*eps*lap_v + (v-1.0)/(2.0*eps) )
        v = np.clip(v - tau_v * dv, 0.0, 1.0)
    return u, v

def quantize_levels(u, levels=3, seed=0):
    H, W = u.shape
    km = KMeans(n_clusters=levels, n_init=10, random_state=seed)
    raw = km.fit_predict(u.reshape(-1,1)).reshape(H, W)
    means = []
    for k in np.unique(raw):
        m = u[raw==k].mean() if np.any(raw==k) else -np.inf
        means.append((k, m))
    means.sort(key=lambda x: x[1], reverse=True)
    mapping = { old:new for new,(old,_) in enumerate(means) }
    labels = np.vectorize(mapping.get)(raw)
    return labels

def nested_cleanup_from_labels(labels, levels, min_area=128, open_r=1, close_r=1):
    H, W = labels.shape
    if levels <= 1: return labels.copy()
    nested_masks = []; prev = np.ones((H,W), dtype=bool)
    for k in range(1, levels):
        mk = labels >= k
        if open_r > 0:  mk = morphology.opening(mk, morphology.disk(open_r))
        if close_r > 0: mk = morphology.closing(mk, morphology.disk(close_r))
        if min_area > 0:
            mk = morphology.remove_small_objects(mk, min_size=min_area)
            mk = morphology.remove_small_holes(mk, area_threshold=min_area)
        mk = mk & prev; nested_masks.append(mk); prev = mk
    if nested_masks:
        labels_nested = np.stack(nested_masks, axis=-1).sum(axis=-1).astype(int)
    else:
        labels_nested = np.zeros((H,W), dtype=int)
    return labels_nested

def extract_contours_from_nested(labels_nested, levels, simplify='vw', rdp_frac=0.002, vw_area_frac=0.0005):
    H, W = labels_nested.shape
    contours_by_level = {k: [] for k in range(levels)}
    if levels <= 1: return contours_by_level
    if simplify not in ('vw','rdp'): simplify = 'vw'
    eps = rdp_frac * max(H, W)
    vw_thresh = vw_area_frac * (H * W)
    for k in range(1, levels):
        mk = (labels_nested >= k).astype(float)
        cs = measure.find_contours(mk, 0.5)
        sims = []
        for c in cs:
            if len(c) < 3:
                sims.append(c); continue
            sims.append(visvalingam_whyatt(c, area_thresh=vw_thresh, keep_ratio=None, preserve_ends=True)
                        if simplify=='vw' else rdp(c, eps))
        contours_by_level[k] = sims
    return contours_by_level

# ===================== Plotting helpers =====================
def fig_overview(f, u, labels, levels):
    cmap = grayscale_levels_cmap(levels)
    fig, axs = plt.subplots(1,3, figsize=(16,4))
    axs[0].imshow(f, cmap='gray'); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(u, cmap='gray'); axs[1].set_title('AT cartoon u'); axs[1].axis('off')
    axs[2].imshow(labels, vmin=0, vmax=levels-1, cmap=cmap); axs[2].set_title(f'Quantized ({levels}, 0=white)'); axs[2].axis('off')
    return fig

def fig_nested_and_contours(base_img, labels_nested, levels, contours_by_level, simplify):
    cmap = grayscale_levels_cmap(levels)
    colors = _level_colors(levels)
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    axs[0].imshow(labels_nested, vmin=0, vmax=levels-1, cmap=cmap)
    axs[0].set_title('Nested-cleaned labels (0=white)'); axs[0].axis('off')
    axs[1].imshow(base_img, cmap='gray')
    for k in range(1, levels):
        col = colors[k % len(colors)]
        for c in contours_by_level.get(k, []):
            axs[1].plot(c[:,1], c[:,0], color=col)
    axs[1].set_title(f'Contours from cleaned masks (labels ≥ k)  [{simplify}]'); axs[1].axis('off')
    return fig

def render_per_level(base_img, labels_nested, levels, contours_by_level):
    colors = _level_colors(levels)
    figs = []
    for k in range(1, levels):
        mk = (labels_nested >= k)
        fig, axs = plt.subplots(1,2, figsize=(14,5))
        axs[0].imshow(mk, cmap='gray'); axs[0].set_title(f'Mask: labels ≥ {k}'); axs[0].axis('off')
        axs[1].imshow(base_img, cmap='gray')
        col = colors[k % len(colors)]
        for c in contours_by_level.get(k, []):
            axs[1].plot(c[:,1], c[:,0], color=col)
        axs[1].set_title(f'Contours for labels ≥ {k}'); axs[1].axis('off')
        figs.append((k, fig))
    return figs

# ===================== Sidebar (controls) =====================
st.title("Ambrosio–Tortorelli Multi-Level + VW Simplification")

with st.sidebar:
    st.header("Input")
    upl = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp","tif","tiff","webp"])
    url = st.text_input("...or paste image URL (optional)")
    scale = st.slider("Downscale factor (applied before processing)", 0.05, 1.0, 1.0, 0.05)

    st.header("Levels / Quantization")
    levels = st.slider("Levels (0=white ... levels-1=darkest)", 2, 12, 4, 1)
    seed = st.number_input("KMeans random_state", value=0, step=1)

    st.header("Preprocess")
    use_pre = st.checkbox("Enable preprocessing (bilateral + CLAHE)", value=True)
    sigma_c = st.number_input("Bilateral sigma_color", value=0.05, step=0.01, format="%.3f")
    sigma_s = st.number_input("Bilateral sigma_spatial", value=3.0, step=1.0, format="%.1f")
    clahe_clip = st.number_input("CLAHE clip limit", value=0.01, step=0.01, format="%.2f")

    st.header("AT Params")
    at_lambda = st.number_input("lambda", value=0.25, step=0.05, format="%.2f")
    at_alpha  = st.number_input("alpha",  value=0.06, step=0.01, format="%.2f")
    at_eps    = st.number_input("epsilon",value=1.2,  step=0.1,  format="%.1f")
    at_iters  = st.number_input("iterations", value=200, step=10)
    at_tau_u  = st.number_input("tau_u", value=0.2, step=0.05, format="%.2f")
    at_tau_v  = st.number_input("tau_v", value=0.15, step=0.05, format="%.2f")

    st.header("Nested Cleanup")
    min_area = st.number_input("Min area", value=128, step=32)
    open_r   = st.number_input("Opening radius", value=1, step=1)
    close_r  = st.number_input("Closing radius", value=1, step=1)

    st.header("Simplification")
    method   = st.selectbox("Method", ["vw","rdp"], index=0)
    rdp_frac = st.number_input("RDP epsilon (fraction of max(H,W))", value=0.002, step=0.001, format="%.3f")
    vw_frac  = st.number_input("VW area threshold (fraction of image area)", value=0.0005, step=0.0005, format="%.4f")
    stroke_w = st.number_input("SVG stroke width", value=1.0, step=0.5, format="%.1f")

# ===================== Load image =====================
def load_image(upl, url):
    if upl is not None:
        return Image.open(upl).convert("RGB")
    if url:
        try:
            import requests
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load from URL: {e}")
    return None

img = load_image(upl, url)
if img is None:
    st.info("Upload an image or paste a URL to begin.")
    st.stop()

# Downscale first
if scale < 1.0:
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img_proc = img.resize((new_w, new_h), RESAMPLE_LANCZOS)
else:
    img_proc = img

caption(f"Working size: {img_proc.width} × {img_proc.height}")

# ===================== Run pipeline (cached) =====================
@cache_data_deco(show_spinner=True)
def run_pipeline(img_rgb, levels, seed, use_pre, sigma_c, sigma_s, clahe_clip,
                 at_lambda, at_alpha, at_eps, at_iters, at_tau_u, at_tau_v,
                 min_area, open_r, close_r, method, rdp_frac, vw_frac):
    f = to_gray(np.asarray(img_rgb))
    if use_pre:
        f_dn = restoration.denoise_bilateral(f, sigma_color=sigma_c, sigma_spatial=sigma_s)
        f_eq = clahe(f_dn, clip_limit=clahe_clip, nbins=256)
    else:
        f_eq = f

    u, v = ambrosio_tortorelli(f_eq, lam=at_lambda, alpha=at_alpha, eps=at_eps,
                               n_iters=int(at_iters), tau_u=at_tau_u, tau_v=at_tau_v)

    labels = quantize_levels(u, levels=int(levels), seed=int(seed))
    labels_nested = nested_cleanup_from_labels(labels, levels=int(levels),
                                               min_area=int(min_area), open_r=int(open_r), close_r=int(close_r))

    contours = extract_contours_from_nested(labels_nested, levels=int(levels),
                                            simplify=method, rdp_frac=rdp_frac, vw_area_frac=vw_frac)
    return f_eq, u, labels, labels_nested, contours

f_eq, u, labels, labels_nested, contours = run_pipeline(
    img_proc, levels, seed, use_pre, sigma_c, sigma_s, clahe_clip,
    at_lambda, at_alpha, at_eps, at_iters, at_tau_u, at_tau_v,
    min_area, open_r, close_r, method, rdp_frac, vw_frac
)

# ===================== Display =====================
c1, c2 = st.columns([1,1], gap="large")
with c1:
    st.subheader("Original (possibly downscaled)")
    show_image(img_proc)
with c2:
    st.subheader("AT Cartoon u")
    show_image(u, clamp=True)

st.subheader("Overview")
fig = fig_overview(f_eq, u, labels, levels)
show_pyplot(fig)

st.subheader("Nested + Contours")
fig2 = fig_nested_and_contours(f_eq, labels_nested, levels, contours, method)
show_pyplot(fig2)

st.subheader("Per-level Views")
per_level = render_per_level(f_eq, labels_nested, levels, contours)
for k, ffig in per_level:
    show_pyplot(ffig)

# ===================== Downloads =====================
st.header("Download Results")

def array_to_png_bytes(arr, vmin=None, vmax=None, cmap='gray', dpi=150):
    fig = plt.figure()
    ax = plt.axes([0,0,1,1]); ax.axis('off')
    ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# u (PNG)
u_png = array_to_png_bytes(u, vmin=0, vmax=1, cmap='gray', dpi=150)
st.download_button("Download u (PNG)", u_png, file_name="u.png", mime="image/png")

# labels (PNG/NPY)
labels_png = array_to_png_bytes(labels, vmin=0, vmax=levels-1, cmap=grayscale_levels_cmap(levels), dpi=150)
st.download_button("Download labels (PNG)", labels_png, file_name="labels.png", mime="image/png")
labels_npy = io.BytesIO(); np.save(labels_npy, labels); labels_npy.seek(0)
st.download_button("Download labels (NPY)", labels_npy, file_name="labels.npy", mime="application/octet-stream")

# cleaned (PNG/NPY)
clean_png = array_to_png_bytes(labels_nested, vmin=0, vmax=levels-1, cmap=grayscale_levels_cmap(levels), dpi=150)
st.download_button("Download cleaned labels (PNG)", clean_png, file_name="labels_nested.png", mime="image/png")
clean_npy = io.BytesIO(); np.save(clean_npy, labels_nested); clean_npy.seek(0)
st.download_button("Download cleaned labels (NPY)", clean_npy, file_name="labels_nested.npy", mime="application/octet-stream")

# contours JSON (valid JSON) + SVG
def contours_to_jsonable(contours_by_level):
    return {str(k): [c.tolist() for c in conts] for k, conts in contours_by_level.items()}

def contours_to_svg(contours_by_level, width, height, stroke_width=1.0):
    palette = _level_colors(max(1, len(contours_by_level)))
    def rgb255(c): return f'rgb({int(255*c[0])},{int(255*c[1])},{int(255*c[2])})'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    for k in sorted(contours_by_level.keys()):
        col = rgb255(palette[k % len(palette)])
        parts.append(f'<g id="level-{k}" stroke="{col}" fill="none" stroke-width="{stroke_width}">')
        for c in contours_by_level[k]:
            if len(c) == 0: continue
            d = " ".join([f'{c[0,1]:.2f},{c[0,0]:.2f}'] + [f'L {pt[1]:.2f},{pt[0]:.2f}' for pt in c[1:]])
            parts.append(f'<path d="M {d}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    return "\n".join(parts).encode("utf-8")

contours_json_bytes = json.dumps(contours_to_jsonable(contours)).encode("utf-8")
st.download_button("Download contours (JSON)", contours_json_bytes, file_name="contours.json", mime="application/json")

H, W = labels_nested.shape
svg_bytes = contours_to_svg(contours, width=W, height=H, stroke_width=stroke_w)
st.download_button("Download contours (SVG)", svg_bytes, file_name="contours.svg", mime="image/svg+xml")

# ZIP everything
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("u.png", u_png.getvalue())
    zf.writestr("labels.png", labels_png.getvalue())
    zf.writestr("labels.npy", labels_npy.getvalue())
    zf.writestr("labels_nested.png", clean_png.getvalue())
    zf.writestr("labels_nested.npy", clean_npy.getvalue())
    zf.writestr("contours.json", contours_json_bytes)
    zf.writestr("contours.svg", svg_bytes)
zip_buf.seek(0)
st.download_button("Download ALL (zip)", zip_buf, file_name="at_vw_outputs.zip", mime="application/zip")

