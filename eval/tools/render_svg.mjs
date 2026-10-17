// 用 resvg（真实 SVG 渲染器）把 SVG 渲染成 PNG，用于「浏览器一致性」校验
// 用法：node render_svg.mjs <input.svg> <output.png> [width]
import { readFileSync, writeFileSync } from 'node:fs';
import { Resvg } from '@resvg/resvg-js';

const [, , inPath, outPath, widthArg] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: node render_svg.mjs <in.svg> <out.png> [width]');
  process.exit(1);
}

const svg = readFileSync(inPath, 'utf8');
const opts = {
  background: 'rgba(0,0,0,0)',
  fitTo: widthArg ? { mode: 'width', value: Number(widthArg) } : { mode: 'original' },
};
const resvg = new Resvg(svg, opts);
const png = resvg.render().asPng();
writeFileSync(outPath, png);
console.log(JSON.stringify({ out: outPath, width: resvg.width, height: resvg.height }));
