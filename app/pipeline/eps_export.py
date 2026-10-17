"""PostScript Level 3 gradient export for generated SVG paths."""
from __future__ import annotations


def export_eps(svg_text):
    from .export import _parse_svg, _path_to_ps, _hex_to_rgb_floats
    if '<filter' in svg_text or 'opacity=' in svg_text:
        raise ValueError('EPS 不支持此文件的滤镜或透明度，请使用 SVG/PDF')
    width,height,layers=_parse_svg(svg_text)
    if not width or not height:
        raise ValueError('SVG 缺少有效尺寸')
    lines=['%!PS-Adobe-3.0 EPSF-3.0',f'%%BoundingBox: 0 0 {width} {height}',
           '%%LanguageLevel: 3','%%EndComments','gsave',f'0 {height} translate','1 -1 scale']
    for layer in layers:
        if layer.opacity!=1:
            raise ValueError('EPS 无法保留透明度')
        shading=None
        if layer.gradient:
            grad=layer.gradient
            stops=sorted(grad['stops'])
            if stops[0][0]>0:stops.insert(0,(0,stops[0][1]))
            if stops[-1][0]<1:stops.append((1,stops[-1][1]))
            unique={float(offset):color for offset,color in stops}
            stops=sorted(unique.items())
            functions=[]
            for (_,a),(_,b) in zip(stops,stops[1:]):
                ca=' '.join(f'{v:.6f}' for v in _hex_to_rgb_floats(a))
                cb=' '.join(f'{v:.6f}' for v in _hex_to_rgb_floats(b))
                functions.append(f'<< /FunctionType 2 /Domain [0 1] /C0 [{ca}] /C1 [{cb}] /N 1 >>')
            bounds=' '.join(str(offset) for offset,_ in stops[1:-1])
            encode=' '.join('0 1' for _ in functions)
            function=f'<< /FunctionType 3 /Domain [0 1] /Functions [{" ".join(functions)}] /Bounds [{bounds}] /Encode [{encode}] >>'
            if grad.get('type')=='radialGradient':
                kind=3;coords=[grad['cx'],grad['cy'],0,grad['cx'],grad['cy'],grad['r']]
            else:
                kind=2;coords=[grad['x1'],grad['y1'],grad['x2'],grad['y2']]
            shading=f'<< /ShadingType {kind} /ColorSpace /DeviceRGB /Coords [{" ".join(map(str,coords))}] /Function {function} /Extend [true true] >> shfill'
        for d in layer.paths:
            ops=_path_to_ps(d)
            if not ops:continue
            lines.append('gsave newpath')
            lines.extend(ops)
            if shading:
                lines.append('eoclip newpath')
                lines.append(shading)
            elif layer.stroke:
                lines.append(' '.join(f'{v:.6f}' for v in _hex_to_rgb_floats(layer.stroke))+' setrgbcolor')
                lines.append(f'{layer.stroke_width:.6f} setlinewidth 1 setlinecap stroke')
            else:
                lines.append(' '.join(f'{v:.6f}' for v in _hex_to_rgb_floats(layer.fill))+' setrgbcolor eofill')
            lines.append('grestore')
    lines.extend(['grestore','showpage','%%EOF'])
    return ('\n'.join(lines)+'\n').encode('ascii')
