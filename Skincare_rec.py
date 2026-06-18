import json
import html
import os
import re
import io
import base64

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from groq import Groq

st.set_page_config(
    page_title="DermaAssist · AI Skin Intelligence",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --bg:        #F7F4F0;
  --bg2:       #FDFCFA;
  --bg3:       #F0ECE6;
  --bg4:       #E8E2D9;
  --surface:   #FFFFFF;
  --border:    rgba(0,0,0,.07);
  --border2:   rgba(0,0,0,.12);
  --text:      #1A1612;
  --text2:     #5A5249;
  --text3:     #9E9389;

  --teal:      #1B8B7A;
  --teal2:     #156B5E;
  --teal3:     #0F4D44;
  --teal-lt:   #E8F5F3;
  --teal-bdr:  rgba(27,139,122,.2);
  --teal-soft: rgba(27,139,122,.08);

  --rose:      #C94040;
  --rose-lt:   #FDF0F0;
  --rose-bdr:  rgba(201,64,64,.2);

  --amber:     #B07020;
  --amber-lt:  #FDF5E8;
  --amber-bdr: rgba(176,112,32,.2);

  --sage:      #4A7C59;
  --sage-lt:   #EEF4F0;
  --sage-bdr:  rgba(74,124,89,.2);

  --lavender:  #6B5B9E;
  --lav-lt:    #F2F0F8;
  --lav-bdr:   rgba(107,91,158,.2);

  --r3:4px; --r4:6px; --r5:8px; --r6:10px; --r8:14px; --r10:18px; --r12:22px;
  --shadow-sm: 0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
  --shadow:    0 2px 8px rgba(0,0,0,.06), 0 8px 32px rgba(0,0,0,.06);
  --shadow-lg: 0 4px 16px rgba(0,0,0,.08), 0 16px 48px rgba(0,0,0,.08);
}

*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  font-family:'DM Sans',system-ui,sans-serif !important;
  color:var(--text) !important;
  background:var(--bg) !important;
}

[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],footer,#MainMenu { display:none !important; }

.block-container {
  max-width:1280px !important;
  padding:0 28px 100px !important;
  margin:0 auto !important;
}

/* ── NAV ── */
.nav {
  display:flex; align-items:center; justify-content:space-between;
  padding:0 28px; height:62px;
  background:rgba(253,252,250,.96);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  position:sticky; top:0; z-index:9999;
  margin:0 -28px 0;
}
.nav-logo {
  display:flex; align-items:center; gap:10px;
}
.nav-logo-mark {
  width:36px; height:36px; border-radius:10px;
  background:linear-gradient(135deg, var(--teal), var(--teal2));
  display:flex; align-items:center; justify-content:center;
  font-size:18px; box-shadow:0 2px 12px rgba(27,139,122,.3);
}
.nav-logo-text {
  font-family:'Cormorant Garamond',serif;
  font-size:21px; font-weight:600; color:var(--text);
  letter-spacing:-.02em;
}
.nav-logo-text em { font-style:italic; color:var(--teal); }
.nav-links { display:flex; gap:2px; }
.nav-links a {
  padding:7px 16px; border-radius:8px;
  font-size:13.5px; color:var(--text2); font-weight:400;
  text-decoration:none; transition:.15s; letter-spacing:-.01em;
}
.nav-links a:hover { color:var(--text); background:var(--bg3); }
.nav-badge {
  display:flex; align-items:center; gap:7px;
  padding:8px 18px; border-radius:9px;
  background:var(--teal-lt); color:var(--teal);
  border:1px solid var(--teal-bdr);
  font-size:12px; font-weight:500; letter-spacing:-.01em;
}
.pulse-dot { width:6px; height:6px; border-radius:50%; background:var(--teal); animation:pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

/* ── HERO ── */
.hero-wrap {
  margin:0 -28px;
  position:relative; overflow:hidden;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
}
/* Hero background: close-up macro skin texture / skin analysis */
.hero-bg-img {
  position:absolute; inset:0;
  background-image:url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1600&q=80&auto=format&fit=crop');
  background-size:cover; background-position:center;
  opacity:.18;
}
.hero-bg-pattern {
  position:absolute; inset:0;
  background-image:
    radial-gradient(circle at 65% 40%, rgba(27,139,122,.10) 0%, transparent 55%),
    radial-gradient(circle at 15% 75%, rgba(27,139,122,.06) 0%, transparent 40%);
}
.hero-inner {
  max-width:1280px; margin:0 auto; padding:88px 28px 76px;
  display:grid; grid-template-columns:1fr 520px; gap:64px; align-items:center;
  position:relative; z-index:1;
}
.hero-eyebrow {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 14px; border-radius:99px;
  background:var(--teal-lt); border:1px solid var(--teal-bdr);
  font-family:'DM Mono',monospace;
  font-size:10px; font-weight:500; letter-spacing:.1em;
  text-transform:uppercase; color:var(--teal); margin-bottom:26px;
}
.hero-dot { width:5px; height:5px; border-radius:50%; background:var(--teal); animation:pulse 1.8s ease-in-out infinite; }
.hero-h1 {
  font-family:'Cormorant Garamond',serif;
  font-size:clamp(52px,6.5vw,88px);
  font-weight:400; letter-spacing:-.03em; line-height:.9;
  color:var(--text); margin-bottom:24px;
}
.hero-h1 em { font-style:italic; color:var(--teal); }
.hero-h1 span { display:block; }
.hero-p {
  font-size:15.5px; color:var(--text2); line-height:1.85;
  max-width:480px; margin-bottom:38px; font-weight:300;
}
.hero-btns { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.btn-prim {
  padding:14px 28px; border-radius:10px;
  background:var(--teal); color:#fff;
  font-size:14px; font-weight:500; text-decoration:none;
  display:inline-flex; align-items:center; gap:8px;
  transition:.2s; letter-spacing:-.01em;
  box-shadow:0 2px 10px rgba(27,139,122,.3);
}
.btn-prim:hover { background:var(--teal2); transform:translateY(-2px); box-shadow:0 6px 22px rgba(27,139,122,.35); }
.btn-ghost {
  padding:13px 24px; border-radius:10px;
  background:transparent; color:var(--text2);
  font-size:14px; font-weight:400; text-decoration:none;
  border:1.5px solid var(--border2); transition:.16s;
  display:inline-flex; align-items:center; gap:7px; letter-spacing:-.01em;
}
.btn-ghost:hover { color:var(--text); border-color:rgba(0,0,0,.2); background:var(--bg3); }
.hero-stats {
  display:flex; gap:0; border-top:1px solid var(--border);
  margin-top:48px; padding-top:34px;
}
.hstat { flex:1; padding:0 20px; border-right:1px solid var(--border); }
.hstat:first-child { padding-left:0; }
.hstat:last-child { border-right:none; }
.hstat-n {
  font-family:'Cormorant Garamond',serif;
  font-size:40px; font-weight:500; letter-spacing:-.04em;
  color:var(--teal); line-height:1; margin-bottom:5px;
}
.hstat-l { font-size:10.5px; color:var(--text3); letter-spacing:.08em; text-transform:uppercase; font-family:'DM Mono',monospace; }

/* ─── HERO IMAGE GRID (4-quadrant, no overlap) ─── */
.hero-img-grid {
  display:grid;
  grid-template-columns:1fr 1fr;
  grid-template-rows:240px 220px;
  gap:12px;
  width:100%;
}
.hig-cell {
  border-radius:16px; overflow:hidden;
  border:1px solid var(--border); box-shadow:var(--shadow);
  background:var(--surface);
  position:relative;
}
.hig-cell img {
  width:100%; height:100%; object-fit:cover; display:block;
  transition:transform .4s ease;
}
.hig-cell:hover img { transform:scale(1.04); }
.hig-cell.tall { grid-row:span 2; border-radius:20px; }
.hig-label {
  position:absolute; bottom:10px; left:10px;
  padding:5px 11px; border-radius:6px;
  background:rgba(0,0,0,.55); backdrop-filter:blur(8px);
  font-family:'DM Mono',monospace; font-size:9px;
  letter-spacing:.07em; text-transform:uppercase; color:#fff;
}
.hig-badge {
  position:absolute; top:10px; right:10px;
  padding:4px 10px; border-radius:6px;
  background:var(--teal); color:#fff;
  font-family:'DM Mono',monospace; font-size:8.5px;
  letter-spacing:.07em; text-transform:uppercase;
  box-shadow:0 2px 8px rgba(27,139,122,.4);
}

/* ── SECTION HEADER ── */
.sec-hd { padding:56px 0 0; }
.sec-eyebrow {
  display:flex; align-items:center; gap:8px;
  font-family:'DM Mono',monospace; font-size:10px;
  font-weight:500; letter-spacing:.14em; text-transform:uppercase;
  color:var(--teal); margin-bottom:10px;
}
.sec-eyebrow::before { content:''; width:16px; height:1px; background:var(--teal); }
.sec-h {
  font-family:'Cormorant Garamond',serif;
  font-size:clamp(32px,4vw,52px);
  font-weight:400; letter-spacing:-.03em; color:var(--text);
  line-height:.95; margin-bottom:12px;
}
.sec-p { font-size:14px; color:var(--text2); line-height:1.9; max-width:500px; font-weight:300; }
.divider { height:1px; background:var(--border); margin:56px 0 0; }

/* ── HOW IT WORKS STRIP ── */
.how-grid {
  display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border);
  border-radius:16px; overflow:hidden; margin-top:28px;
}
.how-card { background:var(--surface); padding:28px 24px; transition:.15s; }
.how-card:hover { background:var(--bg2); }
.how-num {
  font-family:'Cormorant Garamond',serif;
  font-size:48px; font-weight:400; color:var(--teal-lt);
  line-height:1; margin-bottom:12px; letter-spacing:-.04em;
  -webkit-text-stroke:1.5px var(--teal-bdr);
}
.how-card h4 { font-size:14px; font-weight:600; color:var(--text); margin-bottom:6px; }
.how-card p  { font-size:12.5px; color:var(--text2); line-height:1.75; }
.how-icon { font-size:24px; margin-bottom:10px; display:block; }

/* ── FEATURE CARDS ── */
.feat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:28px; }
.feat-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:20px 18px;
  box-shadow:var(--shadow-sm); transition:.2s;
}
.feat-card:hover { transform:translateY(-3px); border-color:var(--teal-bdr); box-shadow:0 8px 24px rgba(27,139,122,.1); }
.feat-ic { font-size:26px; margin-bottom:12px; display:block; }
.feat-card h4 { font-size:13px; font-weight:600; color:var(--text); margin-bottom:5px; }
.feat-card p  { font-size:11.5px; color:var(--text2); line-height:1.7; }

/* ── SCAN CARD ── */
.scan-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:20px; overflow:hidden;
  box-shadow:var(--shadow); margin-top:28px;
}
.scan-head {
  padding:22px 28px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:14px;
  background:var(--bg2);
}
.scan-icon {
  width:42px; height:42px; border-radius:11px;
  background:var(--teal-lt); border:1px solid var(--teal-bdr);
  display:flex; align-items:center; justify-content:center; font-size:20px;
  flex-shrink:0;
}
.scan-head h3 {
  font-family:'Cormorant Garamond',serif; font-size:22px; font-weight:500;
  color:var(--text); letter-spacing:-.02em; margin-bottom:2px;
}
.scan-head p { font-size:11px; color:var(--text3); font-family:'DM Mono',monospace; letter-spacing:.04em; }

.upload-layout { display:grid; grid-template-columns:1fr 310px; gap:20px; padding:28px; }

[data-testid="stFileUploader"] > div {
  background:var(--bg2) !important;
  border:2px dashed rgba(27,139,122,.25) !important;
  border-radius:14px !important;
  padding:48px 24px !important; transition:.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
  border-color:rgba(27,139,122,.5) !important;
  background:var(--teal-lt) !important;
}
[data-testid="stFileUploader"] label { color:var(--text2) !important; font-size:13px !important; font-family:'DM Sans',sans-serif !important; }

.tips-card {
  background:var(--bg2); border:1px solid var(--border);
  border-radius:14px; padding:22px;
}
.tips-label {
  font-family:'DM Mono',monospace; font-size:9.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--text3); margin-bottom:11px;
}
.tip-row { display:flex; gap:9px; align-items:flex-start; margin-bottom:9px; }
.tip-ic {
  width:18px; height:18px; border-radius:4px;
  display:flex; align-items:center; justify-content:center;
  font-size:9px; font-weight:700; flex-shrink:0; margin-top:1px;
}
.tip-ok  { background:var(--teal-lt); color:var(--teal); border:1px solid var(--teal-bdr); }
.tip-no  { background:var(--rose-lt); color:var(--rose); border:1px solid var(--rose-bdr); }
.tip-row p { font-size:12px; color:var(--text2); line-height:1.55; }
.cond-tags { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:14px; }
.cond-tag {
  padding:4px 11px; border-radius:99px;
  background:var(--surface); border:1px solid var(--border);
  font-size:11px; color:var(--text2); letter-spacing:-.01em;
}

/* ── REPORT ── */
.report-header {
  display:flex; align-items:center; gap:14px;
  padding:18px 28px;
  background:linear-gradient(90deg, var(--teal-lt), transparent);
  border-bottom:1px solid var(--border);
  border-left:3px solid var(--teal);
}
.report-header h4 { font-size:14.5px; font-weight:600; color:var(--text); margin-bottom:3px; }
.report-header p  { font-size:11px; color:var(--text3); font-family:'DM Mono',monospace; letter-spacing:.02em; }
.report-body { padding:28px; display:flex; flex-direction:column; gap:26px; }

.dx-grid { display:grid; grid-template-columns:220px 1fr; gap:18px; }
.img-card {
  border-radius:12px; overflow:hidden;
  border:1px solid var(--border); background:var(--surface);
  box-shadow:var(--shadow-sm);
}
.img-card img { width:100%; height:170px; object-fit:cover; display:block; }
.img-foot { padding:14px; }
.model-chip {
  display:inline-flex; align-items:center; gap:5px;
  padding:2px 9px; border-radius:4px;
  background:var(--teal-lt); border:1px solid var(--teal-bdr);
  font-family:'DM Mono',monospace; font-size:9px;
  font-weight:500; color:var(--teal); letter-spacing:.06em; text-transform:uppercase;
}
.conf-row { margin-top:10px; }
.conf-labels { display:flex; justify-content:space-between; font-size:10.5px; color:var(--text3); margin-bottom:5px; font-family:'DM Mono',monospace; }
.conf-labels strong { color:var(--teal); font-size:11.5px; }
.conf-track { height:4px; background:var(--bg4); border-radius:99px; overflow:hidden; }
.conf-fill { height:100%; border-radius:99px; background:linear-gradient(90deg, var(--teal), #2BC9B5); }
.trgs-wrap { margin-top:12px; }
.trgs-label { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:.09em; text-transform:uppercase; color:var(--text3); margin-bottom:6px; }
.trgs-list { display:flex; flex-wrap:wrap; gap:4px; }
.trg-tag {
  padding:3px 9px; border-radius:99px;
  background:var(--bg3); border:1px solid var(--border);
  font-size:11px; color:var(--text2);
}

.dx-panel {
  border-radius:12px;
  border:1px solid var(--border); background:var(--surface);
  padding:26px 28px; display:flex; flex-direction:column; justify-content:space-between;
  box-shadow:var(--shadow-sm);
}
.sev-pill {
  display:inline-flex; align-items:center; gap:6px;
  padding:4px 12px; border-radius:99px;
  font-family:'DM Mono',monospace;
  font-size:9.5px; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
  width:fit-content; margin-bottom:14px;
}
.sev-common  { background:var(--teal-lt); color:var(--teal); border:1px solid var(--teal-bdr); }
.sev-urgent  { background:var(--rose-lt); color:var(--rose); border:1px solid var(--rose-bdr); }
.sev-chronic { background:var(--amber-lt); color:var(--amber); border:1px solid var(--amber-bdr); }
.sev-benign  { background:var(--sage-lt); color:var(--sage); border:1px solid var(--sage-bdr); }
.dx-name {
  font-family:'Cormorant Garamond',serif;
  font-size:clamp(42px,5.5vw,66px);
  font-weight:400; letter-spacing:-.04em;
  color:var(--text); line-height:.88; margin-bottom:14px;
}
.dx-desc { font-size:13.5px; color:var(--text2); line-height:1.85; font-weight:300; max-width:440px; }
.dx-metrics {
  display:flex; gap:0; border-top:1px solid var(--border);
  margin-top:18px; padding-top:16px;
}
.dxm { flex:1; padding:0 14px; border-right:1px solid var(--border); }
.dxm:first-child { padding-left:0; }
.dxm:last-child { border-right:none; }
.dxm small { display:block; font-family:'DM Mono',monospace; font-size:8.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--text3); margin-bottom:4px; }
.dxm b { font-size:20px; font-weight:600; color:var(--text); letter-spacing:-.03em; }
.dxm b.teal { color:var(--teal); }

.alert-urgent {
  display:flex; gap:13px; align-items:flex-start;
  padding:16px 20px; border-radius:12px;
  background:var(--rose-lt); border:1px solid var(--rose-bdr);
  border-left:3px solid var(--rose);
}
.alert-urgent p { font-size:13px; color:var(--rose); line-height:1.75; }
.alert-urgent p strong { display:block; font-weight:600; margin-bottom:3px; font-size:13.5px; }

/* MEDS */
.meds-note {
  font-size:11.5px; color:var(--text3); padding:11px 15px;
  background:var(--bg3); border-radius:8px;
  border-left:2px solid var(--teal-bdr); margin-bottom:14px;
}
.med-grid { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
.med-card { background:var(--surface); padding:20px 20px; transition:.15s; }
.med-card:hover { background:var(--bg2); }
.med-head { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.med-icon {
  width:32px; height:32px; border-radius:8px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-size:15px;
  background:var(--teal-lt); border:1px solid var(--teal-bdr);
}
.med-name { font-size:13px; font-weight:600; color:var(--text); letter-spacing:-.01em; }
.med-type { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:.05em; text-transform:uppercase; color:var(--text3); margin-top:1px; }
.med-desc { font-size:12px; color:var(--text2); line-height:1.75; }
.med-note { margin-top:7px; font-size:11px; color:var(--text3); font-style:italic; }
.rx-tag {
  display:inline-flex; align-items:center;
  padding:1px 7px; border-radius:4px; margin-left:5px;
  font-family:'DM Mono',monospace; font-size:9px; letter-spacing:.05em;
}
.rx-otc { background:var(--sage-lt); color:var(--sage); border:1px solid var(--sage-bdr); }
.rx-rx  { background:var(--lav-lt); color:var(--lavender); border:1px solid var(--lav-bdr); }
.rx-urg { background:var(--rose-lt); color:var(--rose); border:1px solid var(--rose-bdr); }

/* CARE PLAN */
.care-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
.care-card { background:var(--surface); padding:20px 20px; transition:.15s; }
.care-card:hover { background:var(--bg2); }
.care-card.urg { background:var(--rose-lt); border-left:2px solid var(--rose); grid-column:1/-1; }
.care-step {
  display:flex; align-items:center; gap:7px;
  font-family:'DM Mono',monospace; font-size:9px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--teal);
  margin-bottom:8px;
}
.care-card.urg .care-step { color:var(--rose); }
.care-step-ic {
  width:22px; height:22px; border-radius:5px;
  display:flex; align-items:center; justify-content:center;
  font-size:11px; background:var(--teal-lt); border:1px solid var(--teal-bdr);
}
.care-card.urg .care-step-ic { background:var(--rose-lt); border-color:var(--rose-bdr); }
.care-card h4 { font-size:13px; font-weight:600; color:var(--text); margin-bottom:4px; }
.care-card p  { font-size:12px; color:var(--text2); line-height:1.7; }

/* TIMELINE */
.tl-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.tl-card {
  padding:20px 16px; border-radius:12px;
  background:var(--surface); border:1px solid var(--border);
  text-align:center; transition:.2s; box-shadow:var(--shadow-sm);
}
.tl-card:hover { transform:translateY(-3px); border-color:var(--teal-bdr); box-shadow:0 8px 24px rgba(27,139,122,.1); }
.tl-icon { font-size:24px; display:block; margin-bottom:10px; }
.tl-card h5 {
  font-family:'DM Mono',monospace; font-size:9.5px;
  color:var(--teal); font-weight:500; letter-spacing:.06em;
  text-transform:uppercase; margin-bottom:6px;
}
.tl-card p { font-size:11px; color:var(--text2); line-height:1.7; }

/* SOURCES */
.src-panel { background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:16px 20px; }
.src-head { font-family:'DM Mono',monospace; font-size:9.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--text3); margin-bottom:10px; display:flex; align-items:center; gap:7px; }
.src-chips { display:flex; flex-wrap:wrap; gap:6px; }
.src-chip {
  padding:5px 12px; border-radius:6px;
  background:var(--surface); border:1px solid var(--border);
  font-size:11.5px; color:var(--text2);
  text-decoration:none; transition:.13s;
}
.src-chip:hover { border-color:var(--teal-bdr); color:var(--teal); background:var(--teal-lt); }

/* ── CHAT ── */
.chat-wrap { background:var(--surface); border:1px solid var(--border); border-radius:20px; overflow:hidden; box-shadow:var(--shadow); }
.chat-head {
  padding:18px 24px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:13px; background:var(--bg2);
}
.chat-avatar {
  width:40px; height:40px; border-radius:50%;
  background:linear-gradient(135deg, var(--teal-lt), var(--teal-bdr));
  border:1.5px solid var(--teal-bdr);
  display:flex; align-items:center; justify-content:center; font-size:19px;
}
.chat-head h3 { font-family:'Cormorant Garamond',serif; font-size:19px; font-weight:500; color:var(--text); margin-bottom:2px; letter-spacing:-.02em; }
.chat-head p  { font-size:10.5px; color:var(--text3); font-family:'DM Mono',monospace; letter-spacing:.02em; }
.chat-body { padding:18px 22px; }
.msgs { max-height:290px; overflow-y:auto; display:flex; flex-direction:column; gap:9px; margin-bottom:14px; }
.msgs::-webkit-scrollbar { width:2px; }
.msgs::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:99px; }
.msg { max-width:82%; font-size:13px; line-height:1.75; padding:11px 15px; border-radius:12px; }
.msg.user { align-self:flex-end; background:var(--teal); color:#fff; font-weight:400; border-bottom-right-radius:3px; }
.msg.bot  { align-self:flex-start; background:var(--bg2); color:var(--text); border:1px solid var(--border); border-bottom-left-radius:3px; }
.chat-empty {
  text-align:center; padding:32px 20px; background:var(--bg2);
  border-radius:12px; border:1px dashed var(--border2); color:var(--text3);
  font-size:13px; margin-bottom:14px;
}
.chat-empty span { font-size:28px; display:block; margin-bottom:8px; }
.qbtns { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:11px; }
.qbtn { padding:6px 13px; border-radius:99px; background:var(--bg2); border:1px solid var(--border); color:var(--text2); font-size:11.5px; cursor:pointer; transition:.14s; }
.qbtn:hover { border-color:var(--teal-bdr); color:var(--teal); background:var(--teal-lt); }

/* ── CLINIC ── */
.clinic-card { background:var(--surface); border:1px solid var(--border); border-radius:20px; overflow:hidden; box-shadow:var(--shadow); }
.clinic-head { padding:20px 26px; border-bottom:1px solid var(--border); display:flex; gap:13px; align-items:center; background:var(--bg2); }
.clinic-ic { width:38px; height:38px; border-radius:10px; background:var(--teal-lt); border:1px solid var(--teal-bdr); display:flex; align-items:center; justify-content:center; font-size:17px; }
.clinic-head h3 { font-family:'Cormorant Garamond',serif; font-size:20px; font-weight:500; color:var(--text); margin-bottom:2px; }
.clinic-head p  { font-size:11.5px; color:var(--text3); }
.clinic-result { display:flex; gap:11px; padding:13px 14px; border-radius:10px; background:var(--bg2); border:1px solid var(--border); margin-bottom:8px; transition:.14s; }
.clinic-result:hover { border-color:var(--teal-bdr); background:var(--teal-lt); }
.cr-icon { width:34px; height:34px; border-radius:8px; background:var(--surface); border:1px solid var(--border); display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
.clinic-result h4 { font-size:13px; font-weight:500; color:var(--text); margin-bottom:2px; }
.clinic-result p  { font-size:11px; color:var(--text3); }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stTextInput"] input {
  background:var(--bg2) !important; border:1px solid var(--border2) !important;
  border-radius:9px !important; color:var(--text) !important;
  padding:10px 14px !important; font-family:'DM Sans',sans-serif !important;
  font-size:13px !important;
}
[data-testid="stTextInput"] input::placeholder { color:var(--text3) !important; }
[data-testid="stTextInput"] input:focus {
  border-color:var(--teal-bdr) !important;
  box-shadow:0 0 0 3px rgba(27,139,122,.08) !important;
  background:var(--surface) !important; outline:none !important;
}
[data-testid="stTextInput"] label { color:var(--text2) !important; font-size:11.5px !important; }

.stButton > button {
  width:100% !important; min-height:40px !important;
  border-radius:9px !important; border:1.5px solid var(--border2) !important;
  background:var(--surface) !important; color:var(--text) !important;
  font-weight:500 !important; font-size:13px !important;
  font-family:'DM Sans',sans-serif !important; transition:all .16s !important;
  box-shadow:var(--shadow-sm) !important;
}
.stButton > button:hover {
  background:var(--teal) !important; color:#fff !important;
  border-color:var(--teal) !important;
  box-shadow:0 4px 14px rgba(27,139,122,.25) !important;
  transform:translateY(-1px) !important;
}

[data-testid="stProgress"] > div { background:var(--bg4) !important; border-radius:99px !important; height:4px !important; }
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg, var(--teal), #2BC9B5) !important; border-radius:99px !important; }
[data-testid="stAlert"] { background:var(--bg2) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text) !important; }
[data-testid="stFileUploaderDropzoneInput"] { display:none !important; }

/* ── FOOTER ── */
.footer { margin:72px -28px 0; border-top:1px solid var(--border); background:var(--bg2); }
.footer-in { max-width:1280px; margin:0 auto; padding:48px 28px 32px; display:grid; grid-template-columns:2fr 1fr 1fr; gap:48px; }
.footer-brand {
  font-family:'Cormorant Garamond',serif; font-size:24px;
  font-weight:500; color:var(--text); margin-bottom:12px; letter-spacing:-.02em;
}
.footer-brand em { font-style:italic; color:var(--teal); }
.footer-about { font-size:12.5px; color:var(--text3); line-height:2; }
.footer-col h5 { font-family:'DM Mono',monospace; font-size:9.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--text3); margin-bottom:14px; }
.footer-col a { display:block; font-size:13px; color:var(--text2); text-decoration:none; margin-bottom:9px; transition:.13s; }
.footer-col a:hover { color:var(--teal); }
.footer-bottom { max-width:1280px; margin:0 auto; padding:14px 28px; border-top:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
.footer-copy { font-family:'DM Mono',monospace; font-size:10px; color:var(--text3); }
.footer-disc { font-size:11px; color:var(--text3); max-width:520px; text-align:right; line-height:1.7; }

/* RESPONSIVE */
@media(max-width:1000px) {
  .hero-inner { grid-template-columns:1fr; }
  .hero-img-grid { display:none; }
  .upload-layout { grid-template-columns:1fr; }
  .dx-grid { grid-template-columns:1fr; }
  .care-grid { grid-template-columns:1fr 1fr; }
  .footer-in { grid-template-columns:1fr; }
  .feat-grid { grid-template-columns:1fr 1fr; }
  .how-grid { grid-template-columns:1fr; }
}
@media(max-width:680px) {
  .tl-grid { grid-template-columns:1fr 1fr; }
  .med-grid { grid-template-columns:1fr; }
  .care-grid { grid-template-columns:1fr; }
  .feat-grid { grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)

# ── LOGIC ──────────────────────────────────────────────────────────────────
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Error configuring Groq API: {e}"); st.stop()

@st.cache_resource
def load_and_cache_tflite_model():
    model_path = "skin_model.tflite"
    model_url = "https://huggingface.co/Tanishq77/skin-condition-classifier/resolve/main/skin_model.tflite"
    if not os.path.exists(model_path):
        st.info("Downloading AI model — first-time setup, please wait.")
        try:
            resp = requests.get(model_url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model: {e}"); return None
    try:
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors(); return interp
    except Exception as e:
        st.error(f"Error loading model: {e}"); return None

interpreter = load_and_cache_tflite_model()
if interpreter is None: st.stop()

CLASS_NAMES = ["Acne","Carcinoma","Eczema","Keratosis","Milia","Rosacea"]

MEDICATIONS = {
    "Acne": [
        {"name":"Benzoyl Peroxide","type":"OTC Topical","icon":"🧴","rx":"otc","desc":"2.5–10% gel/wash. Kills C. acnes, reduces inflammation. First-line per AAD 2024 guidelines. Apply once daily, build tolerance.","note":"⚠️ May bleach fabrics. Start low strength (2.5%) to minimise irritation."},
        {"name":"Salicylic Acid","type":"OTC Topical","icon":"💊","rx":"otc","desc":"0.5–2% cleanser or leave-on. Beta-hydroxy acid that exfoliates inside pores. Best for blackheads and mild comedonal acne.","note":"✓ Safe for pregnancy (limited area). Avoid in salicylate allergy."},
        {"name":"Adapalene (Differin)","type":"OTC/Rx Retinoid","icon":"⚗️","rx":"otc","desc":"0.1% gel — OTC retinoid. Comedolytic and anti-inflammatory. Apply pea-size amount nightly. Full effect at 12 weeks.","note":"⚠️ Causes initial purging (weeks 2–6). Use SPF daily. Avoid in pregnancy."},
        {"name":"Tretinoin","type":"Prescription Retinoid","icon":"💉","rx":"rx","desc":"0.025–0.1% cream/gel. Gold-standard retinoid per AAD. Increases cell turnover, clears clogged follicles. Rx required.","note":"⚠️ Rx only. Strict SPF essential. Avoid in pregnancy (Category X)."},
        {"name":"Clindamycin + BPO","type":"Rx Combination","icon":"🔬","rx":"rx","desc":"Fixed-dose combination (e.g. Duac). Antibiotic + BPO prevents resistance. Strongly recommended by AAD 2024 guidelines.","note":"✓ BPO prevents antibiotic resistance. Do not use antibiotics alone."},
        {"name":"Oral Doxycycline","type":"Rx Oral Antibiotic","icon":"💊","rx":"rx","desc":"50–100mg twice daily for moderate-severe acne. Strongly recommended by AAD. Always combine with BPO. 3–6 month course.","note":"⚠️ Rx only. Photosensitivity risk. Take with food. Not for under 8 or pregnancy."},
    ],
    "Carcinoma": [
        {"name":"Urgent Dermatology Referral","type":"First-line Action","icon":"🚨","rx":"urg","desc":"Suspected skin carcinoma (BCC/SCC) requires immediate dermatologist evaluation. Do NOT self-treat. Dermoscopy and biopsy needed.","note":"⚠️ AAD: All suspicious lesions need biopsy within 1–2 weeks for definitive diagnosis."},
        {"name":"Sun Protection SPF 50+","type":"Prevention","icon":"☀️","rx":"otc","desc":"Broad-spectrum mineral SPF 50+ (zinc oxide / titanium dioxide). Prevents further UV damage. Reapply every 2 hours outdoors.","note":"✓ Reduces recurrence risk. FDA-approved claim: UVA+UVB protection."},
        {"name":"Surgical Excision","type":"Gold Standard Rx","icon":"🏥","rx":"rx","desc":"Mohs micrographic surgery achieves 99% cure rate for BCC (JAMA Derm). Preferred for high-risk facial/recurrent lesions.","note":"⚠️ Physician-only procedure. Tissue margins assessed in real-time."},
        {"name":"Imiquimod (Aldara)","type":"Rx Topical","icon":"⚗️","rx":"rx","desc":"5% cream immune-modulator for superficial BCC/AK. Rx only. Apply 5×/week for 6 weeks per NICE guidelines.","note":"⚠️ Causes significant local reaction (redness, crusting) — this confirms efficacy."},
    ],
    "Eczema": [
        {"name":"Emollients / Moisturisers","type":"First-line OTC","icon":"💧","rx":"otc","desc":"Apply thick emollient (Cerave, Eucerin, Vaseline) immediately after bathing while skin is damp. Core of all eczema management per NICE.","note":"✓ Apply ≥2× daily. Choose fragrance-free, hypoallergenic formulas."},
        {"name":"Hydrocortisone 1%","type":"OTC Steroid","icon":"🧴","rx":"otc","desc":"Mild topical corticosteroid for acute flares. Apply thin layer to affected areas 1–2× daily. Limit to 7–14 days on face/skin folds.","note":"⚠️ Do not use on face >7 days. Avoid in broken or infected skin."},
        {"name":"Mometasone (Elocon)","type":"Rx Moderate Steroid","icon":"💊","rx":"rx","desc":"0.1% cream/ointment. Medium-potency corticosteroid for moderate flares on body. Once daily application per NICE guidance.","note":"⚠️ Rx only. Limit use to 4 weeks; taper gradually to prevent rebound."},
        {"name":"Tacrolimus (Protopic)","type":"Rx Calcineurin Inhibitor","icon":"⚗️","rx":"rx","desc":"0.03–0.1% ointment. Non-steroidal Rx for facial/fold eczema. Reduces inflammation without skin-thinning side effects of steroids.","note":"✓ Safe for sensitive areas (face, neck, skin folds). Stings initially."},
        {"name":"Dupilumab (Dupixent)","type":"Rx Biologic","icon":"🔬","rx":"rx","desc":"IL-4/13 blocker injection for moderate-severe atopic dermatitis. FDA-approved. 80% show significant improvement in clinical trials.","note":"⚠️ Rx only — dermatologist-prescribed. Self-injectable every 2 weeks."},
        {"name":"Colloidal Oatmeal","type":"OTC Soothing Agent","icon":"🌿","rx":"otc","desc":"1% colloidal oatmeal (Aveeno) cream/bath. FDA-approved skin protectant. Reduces itch and restores skin barrier via beta-glucan.","note":"✓ Safe for all ages including infants. Fragrance-free formulations preferred."},
    ],
    "Keratosis": [
        {"name":"Sunscreen SPF 50+","type":"Prevention OTC","icon":"☀️","rx":"otc","desc":"Daily broad-spectrum SPF 50+ mineral sunscreen. Primary prevention for actinic keratosis progression. Zinc oxide preferred for UV-A blocking.","note":"✓ AAD: Even on cloudy days. Reapply every 2 hours outdoors."},
        {"name":"Fluorouracil Cream (Efudex)","type":"Rx Topical Chemo","icon":"⚗️","rx":"rx","desc":"5% 5-FU cream. FDA-approved for actinic keratosis. Apply twice daily ×2–4 weeks. Causes significant inflammation (evidence of efficacy).","note":"⚠️ Rx only. Avoid sun during treatment. Do not use in pregnancy."},
        {"name":"Imiquimod 5%","type":"Rx Immune Modifier","icon":"💊","rx":"rx","desc":"Immune response modifier for AK. Apply 2–3× weekly for 16 weeks. Effective for field cancerization (multiple AKs in same area).","note":"⚠️ Rx only. Expect local skin reactions — monitor for superinfection."},
        {"name":"Diclofenac Gel 3%","type":"Rx Anti-inflammatory","icon":"🧴","rx":"rx","desc":"Solaraze 3% gel. NSAID topical for mild-moderate AK. Apply twice daily for 60–90 days. Well tolerated vs fluorouracil.","note":"⚠️ Rx. Avoid in NSAID allergy. Less effective than 5-FU for thick lesions."},
    ],
    "Milia": [
        {"name":"Retinol / Retinoids","type":"OTC-Rx Exfoliant","icon":"⚗️","rx":"otc","desc":"Topical retinol (OTC 0.025–0.3%) or tretinoin (Rx). Promotes cell turnover, dissolves the keratin plug trapping milia within 6–12 weeks.","note":"✓ Start OTC retinol. Upgrade to Rx tretinoin if no response in 8 weeks."},
        {"name":"Salicylic Acid Cleanser","type":"OTC Exfoliant","icon":"🧴","rx":"otc","desc":"2% salicylic acid face wash used daily. Dissolves intercellular lipids and keratin buildup. Prevents new milia formation.","note":"✓ Morning cleanse only. Follow with non-comedogenic moisturiser."},
        {"name":"Lactic Acid 5–10%","type":"OTC AHA Exfoliant","icon":"💧","rx":"otc","desc":"Alpha-hydroxy acid that gently exfoliates stratum corneum. 5–10% lactic acid lotion (AmLactin, CeraVe SA) used 2–3× weekly.","note":"✓ Milder than glycolic acid. Builds tolerance over 2–3 weeks."},
        {"name":"Dermatologist Extraction","type":"Clinical Procedure","icon":"🔬","rx":"rx","desc":"Manual extraction via sterile lancet — the most effective treatment. In-office procedure, often complete resolution in one session.","note":"⚠️ Dermatologist only. Home extraction risks infection and scarring."},
    ],
    "Rosacea": [
        {"name":"Azelaic Acid 15–20%","type":"Rx Topical","icon":"⚗️","rx":"rx","desc":"Finacea 15% gel. Reduces papules, pustules and redness. Anti-inflammatory and antimicrobial. FDA-approved for rosacea. Apply twice daily.","note":"✓ Well-tolerated. Safe in pregnancy. Causes mild initial tingling."},
        {"name":"Metronidazole Cream","type":"Rx Topical Antibiotic","icon":"💊","rx":"rx","desc":"0.75–1% cream/gel (Rozex). First-line topical Rx per ROSCO guidelines. Reduces papules/pustules via anti-inflammatory mechanisms.","note":"⚠️ Rx only. Apply pea-size amount to affected areas once/twice daily."},
        {"name":"Ivermectin 1% (Soolantra)","type":"Rx Topical","icon":"🔬","rx":"rx","desc":"Anti-parasitic cream reducing Demodex mites linked to rosacea. FDA-approved. Superior to metronidazole in head-to-head RCT.","note":"⚠️ Rx only. Apply once daily. Onset in 4 weeks, full effect at 12 weeks."},
        {"name":"Mineral SPF 50+","type":"OTC Prevention","icon":"☀️","rx":"otc","desc":"Zinc oxide-based mineral sunscreen. UV is the strongest rosacea trigger. Daily SPF essential for disease control. Choose tinted formulas.","note":"✓ Zinc oxide also has mild anti-inflammatory effect on skin. Reapply every 2h."},
        {"name":"Niacinamide 4–10%","type":"OTC Barrier Repair","icon":"🧴","rx":"otc","desc":"Reduces flushing and strengthens skin barrier. 4% niacinamide serum used morning and evening. Well-tolerated, no irritation.","note":"✓ OTC. Safe to combine with metronidazole. Anti-inflammatory properties confirmed in RCTs."},
        {"name":"Oral Doxycycline","type":"Rx Anti-inflammatory","icon":"💊","rx":"rx","desc":"Sub-antimicrobial dose 40mg (Oracea) for papulopustular rosacea. Anti-inflammatory, not antibiotic dose. FDA-approved for rosacea.","note":"⚠️ Rx only. Photosensitivity. Not for pregnancy. Used 3–4 month courses."},
    ],
}

CONDITION_INFO = {
    "Acne":{"emoji":"🔴","severity":"Common","sev_cls":"sev-common",
        "description":"Inflammatory pilosebaceous condition. Causes comedones, papules, pustules and in severe cases nodulo-cystic lesions. Driven by C. acnes bacteria, sebum excess, and hormonal factors.",
        "triggers":["Hormonal fluctuations","Stress","High-GI diet","Occlusive cosmetics","C. acnes overgrowth","Certain medications (lithium, steroids)"],
        "when_to_see_doctor":"Consult a dermatologist for moderate-severe, cystic, scarring, or treatment-resistant acne (no response after 8–10 weeks of OTC care). Source: AAD Guidelines 2024.",
        "ref_links":[("AAD Guidelines 2024","https://www.jaad.org/article/S0190-9622(23)03389-3/fulltext"),("DermNet · Acne","https://dermnetnz.org/topics/acne"),("PubMed · Acne treatment","https://pubmed.ncbi.nlm.nih.gov/?term=acne+vulgaris+treatment")],
        "timeline":[
            {"icon":"🧼","title":"Week 1–2","desc":"Begin BPO 2.5% + gentle cleanser. No picking."},
            {"icon":"🎯","title":"Week 3–8","desc":"Add adapalene 0.1% nightly. Expect initial purge."},
            {"icon":"📈","title":"Week 8–12","desc":"Assess progress. Introduce salicylic acid if needed."},
            {"icon":"👩‍⚕️","title":"Week 12+","desc":"Dermatologist: Rx tretinoin, antibiotics, or isotretinoin."}]},
    "Carcinoma":{"emoji":"⚠️","severity":"Urgent — Refer","sev_cls":"sev-urgent",
        "description":"Skin carcinoma includes basal cell (BCC) and squamous cell (SCC) types. BCC is most common; SCC carries metastatic risk. Early biopsy and excision is curative in most cases.",
        "triggers":["Cumulative UV/solar exposure","Fair skin phenotype","Immunosuppression","Chemical carcinogen exposure","Prior radiation therapy","Genetic syndromes (Gorlin, xeroderma)"],
        "when_to_see_doctor":"URGENT — same-week dermatology referral. Any ABCDE-positive lesion requires biopsy. Do not apply any creams. Source: NICE NG12; AAD skin cancer guidelines.",
        "ref_links":[("NICE NG12 Skin Cancer","https://www.nice.org.uk/guidance/ng12"),("DermNet · Skin Carcinoma","https://dermnetnz.org/topics/basal-cell-carcinoma"),("AAD Skin Cancer","https://www.aad.org/public/diseases/skin-cancer")],
        "timeline":[
            {"icon":"🚨","title":"Today","desc":"Urgent: Book dermatology appointment this week."},
            {"icon":"🔬","title":"Week 1–2","desc":"Dermoscopy + clinical exam."},
            {"icon":"📋","title":"Week 2–4","desc":"Biopsy + histopathology report."},
            {"icon":"🏥","title":"Week 4+","desc":"Surgical excision / Mohs / radiotherapy."}]},
    "Eczema":{"emoji":"🟠","severity":"Chronic","sev_cls":"sev-chronic",
        "description":"Atopic dermatitis causes dry, intensely pruritic inflamed skin with flare-remission cycles. Driven by skin barrier defects (filaggrin gene mutations) and immune dysregulation (Th2 pathway).",
        "triggers":["Aeroallergens (dust mites, pollen)","Dry air / low humidity","Harsh detergents / SLS","Synthetic fabrics","Emotional stress","Sweating"],
        "when_to_see_doctor":"Seek care if skin becomes infected (weeping, crusted, fever), deeply cracked/bleeding, or fails emollients + mild steroid after 2 weeks. Source: NICE NG217; ETFAD/EADV guidelines.",
        "ref_links":[("NICE NG217 Eczema","https://www.nice.org.uk/guidance/ng217"),("DermNet · Atopic Dermatitis","https://dermnetnz.org/topics/atopic-dermatitis"),("AAD Eczema","https://www.aad.org/public/diseases/eczema")],
        "timeline":[
            {"icon":"💧","title":"Daily","desc":"Moisturise ≥2× daily with thick emollient. Wet-wrap if severe."},
            {"icon":"🧴","title":"Week 1–4","desc":"Hydrocortisone 1% for flares. Avoid known triggers."},
            {"icon":"🔍","title":"Month 1–3","desc":"Allergy testing, trigger diary, antihistamines at night."},
            {"icon":"👩‍⚕️","title":"Month 3+","desc":"Tacrolimus / dupilumab if not controlled."}]},
    "Keratosis":{"emoji":"🟡","severity":"Monitor","sev_cls":"sev-benign",
        "description":"Actinic (solar) keratosis — rough scaly precancerous patches on sun-exposed skin. Seborrheic keratosis is benign pigmented plaque. Actinic types carry ~1 in 1000/year malignant transformation risk.",
        "triggers":["Cumulative UV exposure","Fair phototype I–II","Age >50","Organ transplant / immunosuppression","Geographic UV latitude"],
        "when_to_see_doctor":"Dermatologist if lesions change (ABCDE), bleed, are tender, or rapidly enlarge. Annual full-body skin check recommended for all AK patients. Source: JAAD Guidelines 2021.",
        "ref_links":[("JAAD AK Guidelines 2021","https://www.jaad.org/article/S0190-9622(21)00373-3/fulltext"),("DermNet · Actinic Keratosis","https://dermnetnz.org/topics/actinic-keratosis"),("AAD Skin Cancer","https://www.aad.org/public/diseases/skin-cancer")],
        "timeline":[
            {"icon":"☀️","title":"Now","desc":"Daily broad-spectrum SPF 50+, sun-protective clothing."},
            {"icon":"📸","title":"Monthly","desc":"Photograph and self-monitor all lesions."},
            {"icon":"🔬","title":"Every 3–6 mo","desc":"Check for changes, crusting, bleeding, or enlargement."},
            {"icon":"👩‍⚕️","title":"Annually","desc":"Full-body skin check with dermatologist. Consider field therapy."}]},
    "Milia":{"emoji":"🟢","severity":"Benign","sev_cls":"sev-benign",
        "description":"Keratin-filled epidermal cysts (1–2mm white papules) trapped beneath the skin surface. Primary milia arise spontaneously; secondary milia follow trauma, burns, or blistering disorders. Self-resolving in infants.",
        "triggers":["Heavy occlusive skincare","Sun damage (secondary milia)","Skin trauma or burns","Dermabrasion / laser","Blister-forming conditions"],
        "when_to_see_doctor":"Review if milia persist beyond 3 months, multiply rapidly, are unusually large (>3mm), or associated with genetic syndromes in infants. Source: DermNet NZ, AAD.",
        "ref_links":[("DermNet · Milia","https://dermnetnz.org/topics/milia"),("AAD Milia","https://www.aad.org/public/diseases/bumps-and-growths/milia")],
        "timeline":[
            {"icon":"🚫","title":"Immediately","desc":"Discontinue all heavy occlusive creams."},
            {"icon":"🧴","title":"Week 1–4","desc":"Begin retinol + salicylic acid cleanser regimen."},
            {"icon":"📅","title":"Month 1–3","desc":"Most primary milia self-resolve. Continue retinoids."},
            {"icon":"👩‍⚕️","title":"Month 3+","desc":"Dermatologist sterile-lancet extraction if persistent."}]},
    "Rosacea":{"emoji":"🩷","severity":"Chronic","sev_cls":"sev-chronic",
        "description":"Chronic vascular and inflammatory condition of central face. Characterised by flushing, persistent erythema, telangiectasia, and papulopustules. Four subtypes per ROSCO 2019 consensus.",
        "triggers":["UV light (strongest trigger)","Spicy foods / hot beverages","Alcohol (especially red wine)","Temperature extremes","Vigorous exercise","Emotional stress","Demodex mites"],
        "when_to_see_doctor":"Dermatologist if ocular symptoms develop (ocular rosacea — risk of corneal damage), if rhinophyma begins, or if topical Rx fails after 12 weeks. Source: ROSCO 2019; AAD.",
        "ref_links":[("ROSCO 2019 Guidelines","https://onlinelibrary.wiley.com/doi/full/10.1111/bjd.18420"),("DermNet · Rosacea","https://dermnetnz.org/topics/rosacea"),("AAD Rosacea","https://www.aad.org/public/diseases/acne-and-rosacea/rosacea")],
        "timeline":[
            {"icon":"📓","title":"Week 1–2","desc":"Trigger diary. Switch to mineral SPF 50+."},
            {"icon":"🧴","title":"Week 2–6","desc":"Azelaic acid 15% gel + niacinamide serum."},
            {"icon":"🔍","title":"Month 1–3","desc":"Identify and eliminate individual triggers."},
            {"icon":"👩‍⚕️","title":"Month 3+","desc":"Rx: Metronidazole / ivermectin / oral doxycycline 40mg."}]},
}

def get_prediction(image):
    inp = interpreter.get_input_details(); out = interpreter.get_output_details()
    img = image.resize((inp[0]["shape"][1], inp[0]["shape"][2]))
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(
        np.expand_dims(np.array(img).astype("float32"), 0))
    interpreter.set_tensor(inp[0]["index"], arr); interpreter.invoke()
    preds = interpreter.get_tensor(out[0]["index"])[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]), preds

def get_groq_response(prompt, system_prompt=None):
    try:
        msgs = ([{"role":"system","content":system_prompt}] if system_prompt else []) + [{"role":"user","content":prompt}]
        return groq_client.chat.completions.create(
            messages=msgs, model="llama-3.1-8b-instant").choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def parse_json_block(raw):
    try: return json.loads(raw)
    except: pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None

def clean_value(value):
    if isinstance(value, list): return " ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict): return " ".join(f"{k}: {v}" for k,v in value.items())
    return str(value).strip()

def build_fallback_care_plan(prediction, info):
    return {
        "cleanse": "Use a gentle, fragrance-free, pH-balanced cleanser twice daily. Pat skin dry with a clean towel.",
        "treat": f"Apply the first-line evidence-based topical treatment for {prediction.lower()}. Begin with lowest effective concentration.",
        "protect": "Apply a non-comedogenic broad-spectrum SPF 30–50+ mineral sunscreen every morning, regardless of weather.",
        "avoid": f"Avoid picking, scratching, or aggressive exfoliation. Identify and eliminate personal {prediction.lower()} triggers.",
        "monitor": "Photograph the affected area now and track changes every 7–14 days. Note: size, colour, texture, and new symptoms.",
        "doctor": info.get("when_to_see_doctor","Consult a Board-certified dermatologist if symptoms worsen, persist beyond 3–4 weeks, or affect quality of life."),
    }

def get_structured_care_plan(prediction, info):
    prompt = f"""You are a clinical dermatology assistant. Condition detected: {prediction}.
Return ONLY valid JSON, no markdown, no preamble:
{{"cleanse":"one concrete clinical sentence","treat":"one concrete clinical sentence","protect":"one concrete clinical sentence","avoid":"one concrete clinical sentence","monitor":"one concrete clinical sentence","doctor":"one concrete clinical sentence"}}"""
    raw = get_groq_response(prompt)
    parsed = parse_json_block(raw)
    fallback = build_fallback_care_plan(prediction, info)
    if not isinstance(parsed, dict): return fallback
    return {k: clean_value(parsed.get(k, v)) or v for k, v in fallback.items()}

def scrape_with_api(city: str):
    try:
        resp = requests.post("https://www.justdial.com/api/search",
            headers={"User-Agent":"Mozilla/5.0","Referer":f"https://www.justdial.com/{city.lower().strip()}/Dermatologists"},
            json={"search":"Dermatologists","location":city.lower().strip()}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            return [{"name":c.get("name","N/A"),"location":c.get("address","N/A"),"contact":c.get("contact_number","N/A")}
                    for c in data["results"]]
    except Exception as e:
        st.error(f"Failed to fetch clinic data: {e}")
    return None

for k,v in {"chat_history":[],"analysis_done":False,"prediction":None,
            "confidence":None,"care_plan":None,"image_bytes":None}.items():
    if k not in st.session_state: st.session_state[k] = v

# ════════════════════════════════════════════════════════
#  NAV
# ════════════════════════════════════════════════════════
st.markdown("""
<nav class="nav">
  <div class="nav-logo">
    <div class="nav-logo-mark">🫧</div>
    <div class="nav-logo-text">Derma<em>Assist</em></div>
  </div>
  <div class="nav-links">
    <a href="#scan">Scan</a>
    <a href="#chat">AI Chat</a>
    <a href="#doctors">Find Clinic</a>
  </div>
  <div class="nav-badge"><div class="pulse-dot"></div>AI Model Active</div>
</nav>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  HERO  — all images = skin analysis / dermatoscopy related
#
#  Image sources (all free Unsplash, no attribution required):
#  hero-bg:  macro skin texture close-up
#  hig tall: dermoscopy / skin lesion examination tool
#  hig top-right: close-up human skin texture / pores
#  hig btm-right: dermatology lab / pathology slide analysis
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-bg-img"></div>
  <div class="hero-bg-pattern"></div>
  <div class="hero-inner">
    <div>
      <div class="hero-eyebrow"><div class="hero-dot"></div>AI-Powered Dermatological Screening · EfficientNetV2</div>
      <h1 class="hero-h1">
        <span>Your skin,</span>
        <em>understood.</em>
      </h1>
      <p class="hero-p">Upload a close-up photo of any skin concern and receive an AI-powered differential diagnosis, evidence-based medication guide, and structured care protocol — referenced from AAD, NICE, DermNet NZ, and ROSCO guidelines.</p>
      <div class="hero-btns">
        <a href="#scan" class="btn-prim">Start Free Scan →</a>
        <a href="#doctors" class="btn-ghost">🏥 Find a Dermatologist</a>
      </div>
      <div class="hero-stats">
        <div class="hstat"><div class="hstat-n">6</div><div class="hstat-l">Conditions</div></div>
        <div class="hstat"><div class="hstat-n">~87%</div><div class="hstat-l">Accuracy</div></div>
        <div class="hstat"><div class="hstat-n">&lt;5s</div><div class="hstat-l">Analysis</div></div>
        <div class="hstat"><div class="hstat-n">0</div><div class="hstat-l">Data Stored</div></div>
      </div>
    </div>
    <div class="hero-img-grid">
      <!-- TALL LEFT: dermoscopy device being used on skin -->
      <div class="hig-cell tall">
        <img src="https://images.unsplash.com/photo-1584515933487-779824d29309?w=600&q=85&auto=format&fit=crop" alt="Dermoscopy skin lesion examination" loading="lazy" />
        <div class="hig-label">Dermoscopic Examination</div>
        <div class="hig-badge">&#10003; AI Analysed</div>
      </div>
      <!-- TOP RIGHT: extreme close-up human skin texture / pores -->
      <div class="hig-cell">
        <img src="https://images.unsplash.com/photo-1616394584738-fc6e612e71b9?w=500&q=80&auto=format&fit=crop" alt="Close-up skin texture and pores" loading="lazy" />
        <div class="hig-label">Skin Texture Analysis</div>
      </div>
      <!-- BOTTOM RIGHT: dermatology / microscope / lab slide -->
      <div class="hig-cell">
        <img src="https://images.unsplash.com/photo-1579154204601-01588f351e67?w=500&q=80&auto=format&fit=crop" alt="Dermatology lab pathology analysis" loading="lazy" />
        <div class="hig-label">Pathology · AI Classification</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-hd">
  <div class="sec-eyebrow">How It Works</div>
  <h2 class="sec-h">Three steps to clarity</h2>
  <p class="sec-p">From photo to personalised clinical report in under five seconds.</p>
</div>
<div class="how-grid">
  <div class="how-card">
    <span class="how-icon">📷</span>
    <div class="how-num">01</div>
    <h4>Upload your image</h4>
    <p>Take a clear, well-lit close-up of your skin concern — no filter, no makeup. JPG or PNG, processed securely on-device.</p>
  </div>
  <div class="how-card">
    <span class="how-icon">🧠</span>
    <div class="how-num">02</div>
    <h4>AI analyses in seconds</h4>
    <p>Our EfficientNetV2 model trained on dermatology datasets screens for 6 conditions and outputs confidence scores instantly.</p>
  </div>
  <div class="how-card">
    <span class="how-icon">📋</span>
    <div class="how-num">03</div>
    <h4>Receive your report</h4>
    <p>Get a detailed report with differential diagnosis, evidence-based medications, 6-step care protocol, and recovery timeline.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  FEATURES
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-hd">
  <div class="sec-eyebrow">Platform Features</div>
  <h2 class="sec-h">Everything you need</h2>
</div>
<div class="feat-grid">
  <div class="feat-card">
    <span class="feat-ic">🔬</span>
    <h4>EfficientNetV2 Model</h4>
    <p>State-of-the-art TFLite model achieving ~87% accuracy across 6 dermatological conditions.</p>
  </div>
  <div class="feat-card">
    <span class="feat-ic">💊</span>
    <h4>Evidence-based Meds</h4>
    <p>AAD 2024, NICE, and ROSCO guideline-referenced OTC and Rx treatment options with dosing.</p>
  </div>
  <div class="feat-card">
    <span class="feat-ic">🤖</span>
    <h4>AI Clinical Chat</h4>
    <p>Ask condition-specific questions answered by Groq Llama 3.1 with dermatology context.</p>
  </div>
  <div class="feat-card">
    <span class="feat-ic">🏥</span>
    <h4>Clinic Finder</h4>
    <p>Locate board-certified dermatologists near you for professional evaluation and prescription care.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  SCAN
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-hd" id="scan">
  <div class="sec-eyebrow">AI Skin Analysis</div>
  <h2 class="sec-h">Upload &amp; scan</h2>
  <p class="sec-p">Processed on-device. Never stored. Report includes differential, medications, care plan, and recovery timeline.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="scan-card">', unsafe_allow_html=True)
st.markdown("""
<div class="scan-head">
  <div class="scan-icon">🖼️</div>
  <div>
    <h3>Upload skin image</h3>
    <p>// JPG · JPEG · PNG · Analysed by EfficientNetV2-TFLite</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.analysis_done:
    st.markdown('<div class="upload-layout">', unsafe_allow_html=True)
    st.markdown('<div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload skin image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="tips-card">
  <div class="tips-label">Screened conditions</div>
  <div class="cond-tags">
    <span class="cond-tag">🔴 Acne</span>
    <span class="cond-tag">⚠️ Carcinoma</span>
    <span class="cond-tag">🟠 Eczema</span>
    <span class="cond-tag">🟡 Keratosis</span>
    <span class="cond-tag">🟢 Milia</span>
    <span class="cond-tag">🩷 Rosacea</span>
  </div>
  <div style="height:1px;background:var(--border);margin:14px 0"></div>
  <div class="tips-label">Photo guidelines</div>
  <div class="tip-row"><div class="tip-ic tip-ok">✓</div><p>Natural diffuse light, no harsh shadows</p></div>
  <div class="tip-row"><div class="tip-ic tip-ok">✓</div><p>Sharp focus, 15–20 cm from skin</p></div>
  <div class="tip-row"><div class="tip-ic tip-ok">✓</div><p>No filters, no makeup on affected area</p></div>
  <div class="tip-row"><div class="tip-ic tip-no">✗</div><p>No blurry, backlit, or heavily cropped images</p></div>
</div>
</div>
""", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Running EfficientNetV2 model + building evidence-based care plan…"):
            prediction, confidence, _all = get_prediction(image)
            info = CONDITION_INFO.get(prediction, {})
            care_plan = get_structured_care_plan(prediction, info)
            st.session_state.update({
                "image_bytes": uploaded_file.getvalue(),
                "prediction": prediction, "confidence": confidence,
                "care_plan": care_plan, "analysis_done": True
            })
        st.rerun()

# ── RESULTS ─────────────────────────────────────────────
if st.session_state.analysis_done:
    image      = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")
    prediction = st.session_state.prediction
    confidence = float(st.session_state.confidence or 0)
    info       = CONDITION_INFO.get(prediction, {})
    care_plan  = st.session_state.care_plan or build_fallback_care_plan(prediction, info)
    is_urgent  = prediction == "Carcinoma"
    meds       = MEDICATIONS.get(prediction, [])

    buf = io.BytesIO()
    thumb = image.copy(); thumb.thumbnail((600,600))
    thumb.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()

    st.markdown("""<div class="report-header">
  <span style="font-size:24px">✅</span>
  <div>
    <h4>Analysis complete · Clinical report generated</h4>
    <p>// Differential · Evidence-based medications · Care protocol · Recovery timeline</p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="report-body">', unsafe_allow_html=True)

    # 1. DIAGNOSIS
    trigs_html = "".join(f'<span class="trg-tag">{html.escape(t)}</span>' for t in info.get("triggers",[]))
    st.markdown(f"""
<div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);margin-bottom:14px;display:flex;align-items:center;gap:8px"><span style="width:14px;height:1px;background:var(--teal);display:inline-block"></span>Differential Diagnosis</div>
  <div class="dx-grid">
    <div class="img-card">
      <img src="data:image/jpeg;base64,{b64}" alt="uploaded image"/>
      <div class="img-foot">
        <div class="model-chip">EfficientNetV2 · TFLite</div>
        <div class="conf-row">
          <div class="conf-labels"><span>Confidence</span><strong>{confidence*100:.1f}%</strong></div>
          <div class="conf-track"><div class="conf-fill" style="width:{confidence*100:.1f}%"></div></div>
        </div>
        <div class="trgs-wrap">
          <div class="trgs-label">Known triggers</div>
          <div class="trgs-list">{trigs_html}</div>
        </div>
      </div>
    </div>
    <div class="dx-panel">
      <div>
        <div class="sev-pill {info.get('sev_cls','sev-common')}">{html.escape(info.get('emoji',''))} {html.escape(info.get('severity',''))}</div>
        <div class="dx-name">{html.escape(prediction)}</div>
        <p class="dx-desc">{html.escape(info.get('description',''))}</p>
      </div>
      <div class="dx-metrics">
        <div class="dxm"><small>Confidence</small><b class="teal">{confidence*100:.1f}%</b></div>
        <div class="dxm"><small>Model</small><b>ENetV2</b></div>
        <div class="dxm"><small>Status</small><b>{'⚠ URGENT' if is_urgent else '✓ Screened'}</b></div>
        <div class="dxm"><small>Class</small><b>{html.escape(info.get('severity',''))}</b></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # 2. URGENT ALERT
    if is_urgent:
        st.markdown(f"""<div class="alert-urgent">
  <span style="font-size:24px;flex-shrink:0">🚨</span>
  <p><strong>Immediate medical evaluation required</strong>{html.escape(info.get('when_to_see_doctor',''))}</p>
</div>""", unsafe_allow_html=True)

    # 3. MEDICATIONS
    if meds:
        rx_tag = {"otc":'<span class="rx-tag rx-otc">OTC</span>',
                  "rx":'<span class="rx-tag rx-rx">Rx</span>',
                  "urg":'<span class="rx-tag rx-urg">URGENT</span>'}
        med_cards_html = ""
        for m in meds:
            tag = rx_tag.get(m["rx"], "")
            med_cards_html += f"""
<div class="med-card">
  <div class="med-head">
    <div class="med-icon">{html.escape(m['icon'])}</div>
    <div>
      <div class="med-name">{html.escape(m['name'])}{tag}</div>
      <div class="med-type">{html.escape(m['type'])}</div>
    </div>
  </div>
  <div class="med-desc">{html.escape(m['desc'])}</div>
  <div class="med-note">{html.escape(m['note'])}</div>
</div>"""
        st.markdown(f"""<div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <span style="font-size:20px">💊</span>
    <span style="font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:500;color:var(--text);letter-spacing:-.02em">Evidence-Based Treatment Options · {html.escape(prediction)}</span>
    <span style="font-family:'DM Mono',monospace;font-size:9px;padding:2px 9px;border-radius:4px;background:var(--teal-lt);color:var(--teal);border:1px solid var(--teal-bdr);letter-spacing:.06em;text-transform:uppercase">AAD · NICE · DermNet</span>
  </div>
  <div class="meds-note">⚕️ Medications listed are for educational reference only, sourced from AAD 2024, NICE, and DermNet NZ guidelines. Do not start prescription medications without a dermatologist's assessment.</div>
  <div class="med-grid">{med_cards_html}</div>
</div>""", unsafe_allow_html=True)

    # 4. CARE PLAN
    plan_steps = [
        ("🧼","01","Cleanse",      care_plan.get("cleanse",""), False),
        ("🎯","02","Treat",        care_plan.get("treat",""),   False),
        ("🛡️","03","Protect",      care_plan.get("protect",""),False),
        ("🚫","04","Avoid",        care_plan.get("avoid",""),   False),
        ("📈","05","Monitor",      care_plan.get("monitor",""),False),
        ("👩‍⚕️","06","Doctor review",care_plan.get("doctor",""), is_urgent),
    ]
    care_html = ""
    for icon, lbl, title, text, urgent in plan_steps:
        cls = "care-card urg" if urgent else "care-card"
        care_html += f"""<div class="{cls}">
  <div class="care-step"><div class="care-step-ic">{html.escape(icon)}</div>Step {html.escape(lbl)}</div>
  <h4>{html.escape(title)}</h4>
  <p>{html.escape(clean_value(text))}</p>
</div>"""
    st.markdown(f"""<div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <span style="font-size:20px">🗂️</span>
    <span style="font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:500;color:var(--text);letter-spacing:-.02em">6-Step Clinical Care Protocol</span>
    <span style="font-family:'DM Mono',monospace;font-size:9px;padding:2px 9px;border-radius:4px;background:var(--teal-lt);color:var(--teal);border:1px solid var(--teal-bdr);letter-spacing:.06em;text-transform:uppercase">AI · Llama 3.1</span>
  </div>
  <div class="care-grid">{care_html}</div>
</div>""", unsafe_allow_html=True)

    # 5. TIMELINE
    if info.get("timeline"):
        tl_html = "".join(
            f'<div class="tl-card"><span class="tl-icon">{t["icon"]}</span><h5>{html.escape(t["title"])}</h5><p>{html.escape(t["desc"])}</p></div>'
            for t in info["timeline"]
        )
        st.markdown(f"""<div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <span style="font-size:20px">🗓️</span>
    <span style="font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:500;color:var(--text);letter-spacing:-.02em">Recovery Timeline · {html.escape(prediction)}</span>
  </div>
  <div class="tl-grid">{tl_html}</div>
</div>""", unsafe_allow_html=True)

    # 6. SOURCES
    ref_links = info.get("ref_links", [])
    if ref_links:
        chips_html = "".join(
            f'<a href="{html.escape(url)}" target="_blank" class="src-chip">🔗 {html.escape(name)}</a>'
            for name, url in ref_links
        )
        st.markdown(f"""<div class="src-panel">
  <div class="src-head"><span>📚</span>Evidence Sources &amp; Guidelines</div>
  <div class="src-chips">{chips_html}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # report-body

if st.session_state.analysis_done:
    _, c2 = st.columns([5,1])
    with c2:
        if st.button("↺  New scan", use_container_width=True):
            for k in ["analysis_done","prediction","confidence","care_plan","image_bytes","chat_history"]:
                st.session_state[k] = [] if k=="chat_history" else (False if k=="analysis_done" else None)
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # scan-card

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  CHAT + CLINIC
# ════════════════════════════════════════════════════════
if st.session_state.analysis_done:
    prediction = st.session_state.prediction
    confidence = float(st.session_state.confidence or 0)
    info       = CONDITION_INFO.get(prediction, {})

    st.markdown(f"""
<div class="sec-hd" id="chat">
  <div class="sec-eyebrow">AI Dermatology Assistant</div>
  <h2 class="sec-h">Ask about {html.escape(prediction)}</h2>
  <p class="sec-p">Evidence-referenced answers on ingredients, routines, triggers, and when to see a specialist.</p>
</div>
""", unsafe_allow_html=True)

    SYS = (
        f"You are DermaAssist, a clinical dermatology AI assistant. "
        f"Patient's screened condition: {prediction} (AI confidence: {confidence*100:.1f}%). "
        "Provide concise, evidence-based, empathetic guidance. Always cite AAD, NICE, or DermNet NZ where relevant. "
        "Always recommend a Board-certified dermatologist for diagnosis and Rx. "
        "Under 180 words. No bullet lists — flowing prose."
    )

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.markdown(f"""<div class="chat-head">
  <div class="chat-avatar">🤖</div>
  <div>
    <h3>DermaAssist Clinical AI</h3>
    <p>// Llama 3.1 · Evidence-based · {html.escape(prediction)}-focused</p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="chat-body">', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown('<div class="msgs">', unsafe_allow_html=True)
        for m in st.session_state.chat_history:
            cls = "user" if m["role"]=="user" else "bot"
            st.markdown(f'<div class="msg {cls}">{html.escape(m["content"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-empty"><span>💬</span>Ask about medications, triggers, skincare ingredients, or when to see a dermatologist.</div>', unsafe_allow_html=True)

    quick_qs = [
        f"Which OTC products are best for {prediction}?",
        "How long until I see improvement?",
        f"What daily routine should I follow for {prediction}?"
    ]
    st.markdown('<div class="qbtns">', unsafe_allow_html=True)
    cols = st.columns(3); user_in = ""; send = False
    for i, q in enumerate(quick_qs):
        with cols[i]:
            if st.button(q, key=f"q{i}"): user_in = q; send = True
    st.markdown('</div>', unsafe_allow_html=True)

    ci, cb = st.columns([5,1])
    with ci:
        typed = st.text_input("msg", placeholder=f"Ask about {prediction.lower()}…", key="cin", label_visibility="collapsed")
    with cb:
        if st.button("Send →", use_container_width=True): user_in = typed; send = True

    if send and user_in:
        st.session_state.chat_history.append({"role":"user","content":user_in})
        with st.spinner("…"):
            try:
                r = groq_client.chat.completions.create(
                    messages=[{"role":"system","content":SYS}]+st.session_state.chat_history[-8:],
                    model="llama-3.1-8b-instant"
                )
                reply = r.choices[0].message.content
            except Exception as e:
                reply = f"Sorry, couldn't reach the AI: {e}"
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin-top:44px"></div>', unsafe_allow_html=True)

    # CLINIC
    st.markdown("""
<div class="sec-hd" id="doctors">
  <div class="sec-eyebrow">Professional Care</div>
  <h2 class="sec-h">Find a dermatologist</h2>
  <p class="sec-p">AI screening supports early awareness — a Board-certified dermatologist delivers the definitive diagnosis and prescription treatment plan.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div class="clinic-card">
  <div class="clinic-head">
    <div class="clinic-ic">🏥</div>
    <div>
      <h3>Search nearby clinics</h3>
      <p>Board-certified dermatologists in your city</p>
    </div>
  </div>""", unsafe_allow_html=True)

    cc, cb2 = st.columns([4,1])
    with cc:
        city = st.text_input("city", placeholder="e.g. Mumbai, Delhi, Bangalore, Pune…", label_visibility="collapsed")
    with cb2:
        find = st.button("Search →", use_container_width=True, key="clinic_btn")

    if find and city:
        with st.spinner(f"Searching dermatologists in {city}…"):
            clinics = scrape_with_api(city)
        if clinics:
            for c in clinics:
                st.markdown(f"""<div class="clinic-result">
  <div class="cr-icon">🏥</div>
  <div><h4>{html.escape(c['name'])}</h4><p>📍 {html.escape(c['location'])} · 📞 {html.escape(c['contact'])}</p></div>
</div>""", unsafe_allow_html=True)
        else:
            st.warning(f"No results for '{city}'. Try Google Maps: 'Dermatologist near {city}'.")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════
st.markdown("""<div class="footer">
  <div class="footer-in">
    <div>
      <div class="footer-brand">Derma<em>Assist</em></div>
      <p class="footer-about">AI-powered dermatological screening tool built on EfficientNetV2-TFLite and Groq Llama 3.1. Medication recommendations reference AAD 2024 Guidelines, NICE, DermNet NZ, and ROSCO 2019. For educational awareness only. Not a substitute for clinical diagnosis or professional medical advice.</p>
    </div>
    <div class="footer-col">
      <h5>Clinical References</h5>
      <a href="https://www.jaad.org" target="_blank">JAAD · AAD Guidelines</a>
      <a href="https://dermnetnz.org" target="_blank">DermNet NZ</a>
      <a href="https://www.nice.org.uk/guidance/ng12" target="_blank">NICE NG12</a>
      <a href="https://www.aad.org" target="_blank">American Academy of Dermatology</a>
    </div>
    <div class="footer-col">
      <h5>Technology</h5>
      <a href="#">EfficientNetV2 · TFLite</a>
      <a href="#">Groq Llama 3.1</a>
      <a href="#">Streamlit</a>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-copy">© 2025 DermaAssist · Educational screening tool · Not a medical device</div>
    <div class="footer-disc">Always consult a Board-certified dermatologist before starting any treatment. All medication information is for educational reference only.</div>
  </div>
</div>""", unsafe_allow_html=True)