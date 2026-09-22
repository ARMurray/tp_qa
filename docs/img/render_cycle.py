"""Render the cycle diagram to docs/img/pipeline-cycle.png for the README.

    python docs/img/render_cycle.py

Regenerate this whenever the pipeline shape changes. The PNG is committed so
GitHub can show it; this script is committed so the PNG is not an orphan
binary nobody can update.

WHY A PNG AND NOT MERMAID: GitHub renders mermaid natively, but a three-lane
swimlane with edges crossing between lanes hits a layout bug -- flowchart LR
produced a diagram too wide to read once scaled to the column, and flowchart
TB with per-subgraph `direction` failed to render at all. Hand-authored SVG
rasterised to PNG sidesteps the renderer entirely.

WHY IT DOES NOT CLIP: hosted mermaid renderers substitute fonts, measuring
text with one face and drawing it with another, which truncates labels at the
box edge. This restricts itself to faces that ship with Windows, so headless
Chrome measures and draws with the same metrics.

Standalone: literal colours (no CSS variables), and fonts restricted to faces
that ship with Windows, so headless Chrome measures text with the same metrics
it draws with. That is what stops the right-edge clipping the hosted mermaid
renderer produced.
"""
import pathlib, subprocess, os, sys

W, H = 1060, 560
OUT = pathlib.Path(r"C:\Github\tp_qa\docs\img\pipeline-cycle.png")
TMP = pathlib.Path(__file__).resolve().parent / "_fig.tmp.html"

INK, MUTED, LINE, SURF, GROUND = "#12211F", "#5A6B66", "#C8D2CD", "#FFFFFF", "#F7F9F7"
DET, DETBG = "#B4761E", "#F6EEE0"
CORR, CORRBG = "#0E7C6B", "#E6F1EE"
REV, REVBG = "#3D5A99", "#E8ECF5"

SVG = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
    <marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{DET}"/></marker>
    <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{REV}"/></marker>
  </defs>

  <rect x="0" y="0" width="{W}" height="{H}" fill="{GROUND}"/>
  <rect x="8" y="24"  width="1044" height="128" rx="10" fill="{DETBG}"/>
  <rect x="8" y="196" width="1044" height="128" rx="10" fill="{CORRBG}"/>
  <rect x="8" y="368" width="1044" height="128" rx="10" fill="{REVBG}"/>

  <text x="24" y="46"  class="lane" fill="{DET}">detection/ &#183; local</text>
  <text x="24" y="218" class="lane" fill="{CORR}">correction/ &#183; HPC</text>
  <text x="24" y="390" class="lane" fill="{REV}">review_app/ &#183; local</text>

  <g class="n">
    <rect x="130" y="62" width="168" height="52" rx="6"/>
    <text x="214" y="85" class="t">02_extract_tiles</text><text x="214" y="102" class="s">fetch NAIP, cut tiles</text></g>
  <g class="n dash">
    <rect x="352" y="62" width="150" height="52" rx="6"/>
    <text x="427" y="85" class="t">Label Studio</text><text x="427" y="102" class="s">draw boxes by hand</text></g>
  <g class="n">
    <rect x="556" y="62" width="168" height="52" rx="6"/>
    <text x="640" y="85" class="t">04_train_model</text><text x="640" y="102" class="s">YOLOv8s</text></g>
  <g class="n">
    <rect x="778" y="62" width="150" height="52" rx="6" stroke="{DET}" stroke-width="2.25"/>
    <text x="853" y="85" class="t">best.pt</text><text x="853" y="102" class="s">the detector</text></g>

  <g class="n">
    <rect x="76" y="234" width="158" height="52" rx="6"/>
    <text x="155" y="257" class="t">build_training_bins</text><text x="155" y="274" class="s">labels</text></g>
  <g class="n">
    <rect x="272" y="234" width="158" height="52" rx="6"/>
    <text x="351" y="257" class="t">01a 01b 01c 01e</text><text x="351" y="274" class="s">parcels + detection</text></g>
  <g class="n">
    <rect x="468" y="234" width="158" height="52" rx="6"/>
    <text x="547" y="257" class="t">02_feature_engineering</text><text x="547" y="274" class="s">feature tables</text></g>
  <g class="n">
    <rect x="664" y="234" width="158" height="52" rx="6"/>
    <text x="743" y="257" class="t">03 04 06/07 06b/07b</text><text x="743" y="274" class="s">four models</text></g>
  <g class="n">
    <rect x="860" y="234" width="158" height="52" rx="6"/>
    <text x="939" y="257" class="t">05 &#8594; 05b &#8594; 10</text><text x="939" y="274" class="s">infer, rank, queue</text></g>

  <g class="n">
    <rect x="828" y="406" width="190" height="52" rx="6"/>
    <text x="923" y="429" class="t">queue_loader &#8594; app</text><text x="923" y="446" class="s">a human decides</text></g>
  <g class="n">
    <rect x="452" y="406" width="190" height="52" rx="6" stroke="{REV}" stroke-width="2.25"/>
    <text x="547" y="429" class="t">close_round</text><text x="547" y="446" class="s">one command</text></g>
  <g class="n">
    <rect x="76" y="406" width="190" height="52" rx="6"/>
    <text x="171" y="429" class="t">training_locations.gpkg</text><text x="171" y="446" class="s">upload to HPC</text></g>

  <g class="e">
    <line x1="298" y1="88" x2="346" y2="88" marker-end="url(#ah)"/>
    <line x1="502" y1="88" x2="550" y2="88" marker-end="url(#ah)"/>
    <line x1="724" y1="88" x2="772" y2="88" marker-end="url(#ah)"/>
    <line x1="234" y1="260" x2="266" y2="260" marker-end="url(#ah)"/>
    <line x1="430" y1="260" x2="462" y2="260" marker-end="url(#ah)"/>
    <line x1="626" y1="260" x2="658" y2="260" marker-end="url(#ah)"/>
    <line x1="822" y1="260" x2="854" y2="260" marker-end="url(#ah)"/>
    <path d="M 939 286 L 939 400" marker-end="url(#ah)"/>
    <line x1="828" y1="432" x2="648" y2="432" marker-end="url(#ah)"/>
    <line x1="452" y1="432" x2="272" y2="432" marker-end="url(#ah)"/>
  </g>
  <g class="e" stroke="{DET}" stroke-width="2" opacity="1">
    <path d="M 853 114 L 853 166 L 351 166 L 351 228" marker-end="url(#ahd)"/></g>
  <g class="e" stroke="{REV}" stroke-width="2" opacity="1">
    <path d="M 171 406 L 171 344 L 155 344 L 155 292" marker-end="url(#ahr)"/>
    <path d="M 500 406 L 500 344 L 214 344 L 214 120" marker-end="url(#ahr)"/></g>

  <text x="853" y="184" class="el" fill="{DET}">a new best.pt makes every detection output stale &#8212; re-run 01b / 01c / 01e</text>
  <text x="939" y="352" class="el">review_queue_round&#123;N&#125;.parquet</text>
  <text x="738" y="424" class="el">verdicts land in app.db</text>
  <text x="357" y="424" class="el">exports, folds into the master</text>
  <text x="120" y="330" class="el" fill="{REV}">new labels</text>
  <text x="330" y="330" class="el" fill="{REV}">new NAIP tiles for every parcel reviewed</text>
  <text x="530" y="534" class="note">The loop is the point: each pass adds verified locations to the master and annotated tiles to the detector.</text>
</svg>
"""

HTML = f"""<!doctype html><meta charset="utf-8">
<style>
  html,body {{ margin:0; padding:0; background:{GROUND}; }}
  svg {{ display:block; }}
  .n rect {{ fill:{SURF}; stroke:{LINE}; stroke-width:1.5; }}
  .n.dash rect {{ stroke-dasharray:5 4; fill:none; }}
  text {{ font-family:"Segoe UI",Arial,sans-serif; }}
  .t {{ font-family:Consolas,"Courier New",monospace; font-size:12px; fill:{INK}; text-anchor:middle; }}
  .s {{ font-size:10.5px; fill:{MUTED}; text-anchor:middle; }}
  .lane {{ font-family:Consolas,"Courier New",monospace; font-size:11px; letter-spacing:.08em; }}
  .el {{ font-size:10.5px; fill:{MUTED}; text-anchor:middle; }}
  .note {{ font-size:11.5px; fill:{MUTED}; text-anchor:middle; }}
  .e line, .e path {{ stroke:{MUTED}; stroke-width:1.5; fill:none; opacity:.75; }}
</style>
{SVG}
"""
TMP.write_text(HTML, encoding="utf-8")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

CHROME = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
       f"--screenshot={OUT}", f"--window-size={W},{H}",
       "--force-device-scale-factor=2", "--default-background-color=00000000",
       TMP.resolve().as_uri()]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print("exit", r.returncode)
if r.stderr.strip():
    print(r.stderr.strip()[-400:])
if OUT.exists():
    from PIL import Image
    im = Image.open(OUT)
    print(f"PNG {im.size[0]}x{im.size[1]} mode={im.mode} {OUT.stat().st_size/1024:.0f} KB -> {OUT}")
else:
    print("NO PNG PRODUCED")
    sys.exit(1)
