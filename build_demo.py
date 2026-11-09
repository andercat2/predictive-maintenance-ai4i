"""Собирает демо-страницу: подставляет results/demo_data.json в demo/template.html.

demo/predictive_maintenance_demo.html — автономный файл, открывается двойным кликом в любом браузере.
demo/_artifact_body.html               — то же содержимое без обёртки <html>/<head> (для публикации).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tpl = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
data = (ROOT / "results" / "demo_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
body = tpl.replace("/*__DATA__*/", data)

(ROOT / "demo" / "_artifact_body.html").write_text(body, encoding="utf-8")

standalone = (
    '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
    "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
)
out = ROOT / "demo" / "predictive_maintenance_demo.html"
out.write_text(standalone, encoding="utf-8")
print(f"{out} — {out.stat().st_size / 1e6:.2f} MB")
