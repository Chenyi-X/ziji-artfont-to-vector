import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = (await fs.readFile(path.join(root, 'app/static/app.js'), 'utf8'))
  .replace(/init\(\)\.catch\([^\n]+\);\s*$/, '');
class TestForm {
  values = new Map();
  append(key, value) { this.values.set(key, value); }
  get(key) { return this.values.get(key); }
}
const context = vm.createContext({ assert, console, FormData: TestForm, Blob,
  TextEncoder, TextDecoder, performance, setTimeout, clearTimeout, Uint8Array,
  Uint32Array, ArrayBuffer, DataView });
vm.runInContext(source, context);
const archive = await vm.runInContext(`(async () => {
  assert.equal(escapeHtml('<img src=x onerror="x">&'), '&lt;img src=x onerror=&quot;x&quot;&gt;&amp;');
  renderQueue = () => {};
  toast = () => {};
  let current = { preset: 'balanced', tolerance: .5, max_nodes: 500 };
  readParams = () => ({ ...current });
  const requests = [];
  const names = ['same', 'same', 'Same', '_batch_summary', '中文'];
  state.queue = names.map((name, i) => ({ file: { name: name+'.png', webkitRelativePath: 'folder'+i+'/'+name+'.png' }, status: 'idle' }));
  fetch = async (url, request) => {
    requests.push(request.body.get('tolerance'));
    current.tolerance = 9;
    await new Promise(resolve => setTimeout(resolve, 1));
    return { ok: true, json: async () => ({ stem: request.body.get('file').name.slice(0,-4), svg: '<svg>'+('x'.repeat(30000))+'</svg>', metrics: {}, elapsed_ms: 1 }) };
  };
  await runBatch();
  assert.deepEqual(Array.from(requests), [.5,.5,.5,.5,.5]);
  assert(state.queue.every(item => item.params.tolerance === .5));
  assert.equal(paramsEqual({preset:'balanced',glyph_count:4},{preset:'balanced',glyph_count:5}),false);
  let captured;
  download = (blob, filename) => { captured = blob; };
  state.fmt = 'svg';
  await buildZip();
  return new Uint8Array(await captured.arrayBuffer());
})()`, context);
if (process.argv[2]) await fs.writeFile(process.argv[2], archive);
console.log(JSON.stringify({batch_parameter_snapshot:true, duplicate_and_reserved_names:true,
  archive_bytes:archive.length, archive_has_large_central_offset:archive.length>65535}));
