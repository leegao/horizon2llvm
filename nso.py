"""Minimal NSO0 decompressor used by tools/export_shaders.py to scan embedded BNSH shaders."""
import pathlib
import struct


def _lz4_decompress(src: bytes, dec_size: int) -> bytearray:
    out = bytearray(dec_size)
    s, se = 0, len(src)
    d = 0
    while s < se:
        tok = src[s]; s += 1
        lit = tok >> 4
        if lit == 15:
            while s < se:
                b = src[s]; s += 1
                lit += b
                if b != 255:
                    break
        out[d:d + lit] = src[s:s + lit]
        d += lit; s += lit
        if s >= se:
            break
        off = src[s] | (src[s + 1] << 8); s += 2
        ml = tok & 15
        if ml == 15:
            while s < se:
                b = src[s]; s += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        m = d - off
        for _ in range(ml):
            out[d] = out[m]
            d += 1; m += 1
    return out


class Nso:
    def __init__(self, path):
        raw = pathlib.Path(path).read_bytes()
        if raw[:4] != b'NSO0':
            raise ValueError(f'{path}: not an NSO0 file')
        flags = struct.unpack_from('<I', raw, 0x0C)[0]
        segs = [
            (*struct.unpack_from('<III', raw, 0x10 + i * 0x10), struct.unpack_from('<I', raw, 0x60 + i * 4)[0])
            for i in range(3)
        ]
        bss_size = struct.unpack_from('<I', raw, 0x3C)[0]
        size = (segs[2][1] + segs[2][2] + bss_size + 0xFFF) & ~0xFFF
        self.image = bytearray(size)
        for i, (foff, moff, dsz, csz) in enumerate(segs):
            chunk = raw[foff:foff + csz]
            self.image[moff:moff + dsz] = _lz4_decompress(chunk, dsz) if (flags & (1 << i)) else chunk[:dsz]
