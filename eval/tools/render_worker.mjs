import { Resvg } from '@resvg/resvg-js';
import { createInterface } from 'node:readline';
const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  try {
    const request = JSON.parse(line);
    const renderer = new Resvg(request.svg, {
      fitTo: { mode: 'original' },
      background: request.background || 'rgba(0,0,0,0)',
    });
    console.log(JSON.stringify({ png: renderer.render().asPng().toString('base64') }));
  } catch (error) {
    console.log(JSON.stringify({ error: String(error) }));
  }
}
