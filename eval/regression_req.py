"""按原需求记录回归结果，缺测单独列出；报告范围为单张自动检查。"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from app.pipeline.assessment import assess_result
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions


def run_one(path, preset, meta, gradient_required=None):
    data = path.read_bytes()
    truth_path = ROOT/'samples/truth'/path.name
    truth = cv2.imdecode(np.frombuffer(truth_path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE) if truth_path.exists() else None
    text = meta.get('text')
    glyphs = meta.get('glyphs')
    if not isinstance(glyphs, int) or isinstance(glyphs, bool) or glyphs < 1:
        glyphs = sum(1 for c in text if not c.isspace() and c != '/') if text else None
    t0 = time.perf_counter()
    res = convert_bytes(path.name, data, VectorizeOptions.from_preset(preset), with_preview=False,
                        strict_input=False, glyph_count=glyphs, glyph_text=text, truth_mask=truth)
    wall = (time.perf_counter()-t0)*1000
    row = {'file':path.name, 'preset':preset, 'wall_ms':round(wall,1), 'service_ms':round(res.elapsed_ms,1),
           'error':res.error, 'quality':res.quality, 'metrics':res.metrics}
    if res.error:
        row['status']='fail'
        row['items']={'conversion':{'label':'转换完成','status':'fail','note':res.error,'value':None}}
    else:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assessment=assess_result(res.metrics,res.quality,wall,(image.shape[1],image.shape[0]),
                                 gradient_required=gradient_required)
        row.update(assessment)
    return row


def repeatability(path, preset, meta, times):
    data=path.read_bytes()
    hashes=[]
    for _ in range(times):
        res=convert_bytes(path.name,data,VectorizeOptions.from_preset(preset),with_preview=False,
                          strict_input=False,glyph_count=meta.get('glyphs'),glyph_text=meta.get('text'))
        if res.error:
            return {'status':'fail','error':res.error,'requested':times,'completed':len(hashes)}
        hashes.append(hashlib.sha256(res.svg.encode('utf-8')).hexdigest())
    identical=bool(hashes) and len(set(hashes))==1
    return {'status':'pass' if times>=10 and identical else 'untested',
            'requested':times,'completed':len(hashes),'svg_sha256':hashes,'exact_svg_equal':identical,
            'note':'同参数的 SVG 内容完全相同可证明颜色、路径和结构一致；输出不同时需另测各项偏差。'}


def aggregate_status(records):
    statuses = [r.get('status', 'untested') for r in records]
    if 'fail' in statuses:
        return 'fail'
    if not statuses or any(s not in {'pass','not_applicable'} for s in statuses):
        return 'untested'
    return 'pass'


def render_md(report):
    lines=['# 单张自动回归记录','',
           '结果使用通过、失败、未测、不适用四种状态。轮廓与特征分属于诊断指标，视觉及软件兼容性单独列出。',
           '输入不合规的样本保留诊断数据，不能并入合规素材通过率。','',
           '| 样本 | 档位 | 完整调用 ms | 平均节点/字 | 失败项 | 未测项数 | 状态 |',
           '|---|---|---:|---:|---|---:|---|']
    for r in report['rows']:
        fails=[v['label'] for v in r['items'].values() if v['status']=='fail']
        pending=sum(v['status']=='untested' for v in r['items'].values())
        lines.append(f"| {r['file']} | {r['preset']} | {r['wall_ms']} | {r.get('metrics',{}).get('node_per_glyph','')} | {'、'.join(fails)} | {pending} | {r['status']} |")
    lines += ['', '逐条检查、参考口径、输入资格和重复转换记录见 report.json。',
              '百张批量、四浏览器、Illustrator 和首次使用测试仍需独立证据。','']
    return '\n'.join(lines)


def main(argv=None, *, default_out=None, report_name='report'):
    ap=argparse.ArgumentParser()
    ap.add_argument('-o','--out',default=str(default_out or (ROOT/'out/regression')))
    ap.add_argument('--preset',choices=['all','quality','balanced','compact'],default='balanced')
    ap.add_argument('--files',nargs='+')
    ap.add_argument('--labels',help='独立标签 JSON，按文件名映射 gradient_required 布尔值')
    ap.add_argument('--repeat',type=int,default=0)
    ap.add_argument('--repeat-samples',type=int,default=3)
    args=ap.parse_args(argv)
    manifest={m['file']:m for m in json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))}
    labels=json.loads(Path(args.labels).read_text(encoding='utf-8')) if args.labels else {}
    names=args.files or sorted(manifest)
    presets=['quality','balanced','compact'] if args.preset=='all' else [args.preset]
    rows=[]
    for name in names:
        for preset in presets:
            try:
                label=labels.get(name,{}).get('gradient_required')
                if label is not None and not isinstance(label,bool):
                    raise ValueError('gradient_required 必须为布尔值')
                row=run_one(ROOT/'samples/images'/name,preset,manifest.get(name,{}),label)
            except Exception as exc:
                row={'file':name,'preset':preset,'wall_ms':0,'status':'fail','error':str(exc),
                     'items':{'conversion':{'label':'转换完成','status':'fail','note':str(exc)}}}
            rows.append(row)
            print(f"{name} {preset} {row['wall_ms']}ms {row['status']}",flush=True)
    repeats=[]
    if args.repeat:
        for name in names[:args.repeat_samples]:
            for preset in presets:
                repeats.append({'file':name,'preset':preset,**repeatability(ROOT/'samples/images'/name,preset,manifest.get(name,{}),args.repeat)})
    report={'scope':'single_image_checks','rows':rows,'repeat':repeats,
            'status':aggregate_status(rows + repeats),
            'summary':{s:sum(r['status']==s for r in rows) for s in ['pass','fail','untested','not_applicable']},
            'repeat_summary':{s:sum(r['status']==s for r in repeats) for s in ['pass','fail','untested','not_applicable']}}
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    markdown=render_md(report)
    (out/f'{report_name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/f'{report_name}.md').write_text(markdown,encoding='utf-8')
    (out/f'{report_name}.html').write_text('<!doctype html><meta charset="utf-8"><title>单张自动回归记录</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;line-height:1.6}</style><pre>'+html.escape(markdown)+'</pre>',encoding='utf-8')
    return {'fail':1, 'untested':2, 'pass':0}[report['status']]


if __name__=='__main__':
    raise SystemExit(main())
