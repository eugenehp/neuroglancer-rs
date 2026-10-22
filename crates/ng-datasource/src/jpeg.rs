//! The JPEG decoder neuroglancer uses (`src/third_party/jpgjs/jpg.js`, pdf.js's
//! `JpegImage`), ported so decoded chunks match it sample for sample: its
//! integer IDCT, `Int16Array` coefficient stores, nearest-neighbour chroma
//! upsampling and floating-point YCbCr conversion. Other decoders round
//! differently and land ±1 away.
//!
//! JavaScript semantics kept where they decide the output: stores into the
//! coefficient buffers wrap to 16 bits, shifts and bitwise operators first
//! wrap to 32 bits (`ToInt32`), reads past the end of the data give 0, and
//! writes past the end of a buffer are dropped.

use anyhow::{anyhow, bail, Result};

const DCT_ZIG_ZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];
const DCT_COS1: i64 = 4017;
const DCT_SIN1: i64 = 799;
const DCT_COS3: i64 = 3406;
const DCT_SIN3: i64 = 2276;
const DCT_COS6: i64 = 1567;
const DCT_SIN6: i64 = 3784;
const DCT_SQRT2: i64 = 5793;
const DCT_SQRT1D2: i64 = 2896;

/// `ToInt32` of an integer-valued number.
fn i32w(x: i64) -> i32 {
    x as i32
}

/// `x >> s`, as JavaScript does it on a number.
fn sar(x: i64, s: u32) -> i64 {
    (i32w(x) >> s) as i64
}

/// A decoded image: `width × height` pixels, `num_components` interleaved
/// samples each (`getData(width, height, false)`).
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub num_components: usize,
    pub data: Vec<u8>,
}

/// A Huffman tree slot: a symbol, a subtree, or nothing (`undefined`).
#[derive(Clone, Copy)]
enum Slot {
    Empty,
    Value(u8),
    Node(usize),
}

/// Trees as `buildHuffmanTable` makes them: nested arrays, here an arena.
#[derive(Clone)]
struct Huffman {
    nodes: Vec<Vec<Slot>>,
    root: usize,
}

impl Huffman {
    fn build(code_lengths: &[u8; 16], values: &[u8]) -> Result<Self> {
        struct Code {
            children: usize,
            index: usize,
        }
        let mut nodes: Vec<Vec<Slot>> = vec![vec![]];
        let set = |nodes: &mut Vec<Vec<Slot>>, node: usize, index: usize, slot: Slot| {
            let children = &mut nodes[node];
            if children.len() <= index {
                children.resize(index + 1, Slot::Empty);
            }
            children[index] = slot;
        };
        let mut length = 16;
        while length > 0 && code_lengths[length - 1] == 0 {
            length -= 1;
        }
        let mut code = vec![Code { children: 0, index: 0 }];
        let mut k = 0;
        let pop = |code: &mut Vec<Code>| code.pop().ok_or_else(|| anyhow!("invalid huffman table"));
        for (i, &count) in code_lengths.iter().enumerate().take(length) {
            for _ in 0..count {
                let mut p = pop(&mut code)?;
                let value = values.get(k).copied().map_or(Slot::Empty, Slot::Value);
                set(&mut nodes, p.children, p.index, value);
                while p.index > 0 {
                    p = pop(&mut code)?;
                }
                p.index += 1;
                code.push(p);
                while code.len() <= i {
                    nodes.push(vec![]);
                    let q = Code { children: nodes.len() - 1, index: 0 };
                    let p = code.last().unwrap();
                    set(&mut nodes, p.children, p.index, Slot::Node(q.children));
                    code.push(q);
                }
                k += 1;
            }
            if i + 1 < length {
                nodes.push(vec![]);
                let q = Code { children: nodes.len() - 1, index: 0 };
                let p = code.last().unwrap();
                set(&mut nodes, p.children, p.index, Slot::Node(q.children));
                code.push(q);
            }
        }
        // `code[0].children`: the root's array.
        let root = code.first().ok_or_else(|| anyhow!("invalid huffman table"))?.children;
        Ok(Self { nodes, root })
    }
}

struct Component {
    h: usize,
    v: usize,
    /// The table as it was when the frame was read; a missing one acts as
    /// all zeros (every product is `NaN`, and `NaN >> n` is 0).
    quantization_table: [u16; 64],
    blocks_per_line: usize,
    blocks_per_column: usize,
    block_data: Vec<i16>,
    pred: i64,
    huffman_dc: Option<usize>,
    huffman_ac: Option<usize>,
}

impl Component {
    fn get(&self, i: usize) -> i64 {
        self.block_data.get(i).map_or(0, |&v| v as i64)
    }
    fn set(&mut self, i: usize, value: i64) {
        if let Some(slot) = self.block_data.get_mut(i) {
            *slot = value as i16;
        }
    }
}

struct Frame {
    samples_per_line: usize,
    scan_lines: usize,
    progressive: bool,
    max_h: usize,
    max_v: usize,
    mcus_per_line: usize,
    mcus_per_column: usize,
    components: Vec<Component>,
    /// Component id → index.
    component_ids: Vec<Option<usize>>,
}

fn byte(data: &[u8], offset: usize) -> u32 {
    data.get(offset).map_or(0, |&b| b as u32)
}

struct Scan<'a> {
    data: &'a [u8],
    offset: usize,
    bits_data: u32,
    bits_count: u32,
    eobrun: i64,
    successive_ac_state: u8,
    successive_ac_next_value: i64,
    spectral_start: usize,
    spectral_end: usize,
    successive: u32,
}

#[derive(Clone, Copy)]
enum DecodeFn {
    Baseline,
    DcFirst,
    DcSuccessive,
    AcFirst,
    AcSuccessive,
}

impl Scan<'_> {
    fn read_bit(&mut self) -> Result<i64> {
        if self.bits_count > 0 {
            self.bits_count -= 1;
            return Ok(((self.bits_data >> self.bits_count) & 1) as i64);
        }
        self.bits_data = byte(self.data, self.offset);
        self.offset += 1;
        if self.bits_data == 255 {
            let next = byte(self.data, self.offset);
            self.offset += 1;
            if next != 0 {
                bail!("unexpected marker: {:x}", (self.bits_data << 8) | next);
            }
        }
        self.bits_count = 7;
        Ok((self.bits_data >> 7) as i64)
    }

    fn decode_huffman(&mut self, tables: &[Huffman], table: Option<usize>) -> Result<i64> {
        let tree = table.and_then(|t| tables.get(t)).ok_or_else(|| anyhow!("missing huffman table"))?;
        let mut node = tree.root;
        loop {
            let bit = self.read_bit()? as usize;
            match tree.nodes[node].get(bit).copied().unwrap_or(Slot::Empty) {
                Slot::Value(v) => return Ok(v as i64),
                Slot::Node(n) => node = n,
                Slot::Empty => bail!("invalid huffman sequence"),
            }
        }
    }

    fn receive(&mut self, mut length: i64) -> Result<i64> {
        let mut n: i32 = 0;
        while length > 0 {
            n = n.wrapping_shl(1) | self.read_bit()? as i32;
            length -= 1;
        }
        Ok(n as i64)
    }

    fn receive_and_extend(&mut self, length: i64) -> Result<i64> {
        if length == 1 {
            return Ok(if self.read_bit()? == 1 { 1 } else { -1 });
        }
        let n = self.receive(length)?;
        if n >= (1i32.wrapping_shl(length as u32 - 1)) as i64 {
            return Ok(n);
        }
        Ok(n + (-1i32).wrapping_shl(length as u32) as i64 + 1)
    }

    fn decode(&mut self, f: DecodeFn, c: &mut Component, offset: usize, tables: &[Huffman]) -> Result<()> {
        match f {
            DecodeFn::Baseline => {
                let t = self.decode_huffman(tables, c.huffman_dc)?;
                let diff = if t == 0 { 0 } else { self.receive_and_extend(t)? };
                c.pred += diff;
                c.set(offset, c.pred);
                let mut k = 1;
                while k < 64 {
                    let rs = self.decode_huffman(tables, c.huffman_ac)?;
                    let (s, r) = (rs & 15, rs >> 4);
                    if s == 0 {
                        if r < 15 {
                            break;
                        }
                        k += 16;
                        continue;
                    }
                    k += r as usize;
                    let value = self.receive_and_extend(s)?;
                    if let Some(&z) = DCT_ZIG_ZAG.get(k) {
                        c.set(offset + z as usize, value);
                    }
                    k += 1;
                }
            }
            DecodeFn::DcFirst => {
                let t = self.decode_huffman(tables, c.huffman_dc)?;
                let diff = if t == 0 { 0 } else { (i32w(self.receive_and_extend(t)?) << self.successive) as i64 };
                c.pred += diff;
                c.set(offset, c.pred);
            }
            DecodeFn::DcSuccessive => {
                let bit = (self.read_bit()? as i32) << self.successive;
                c.set(offset, (c.get(offset) as i32 | bit) as i64);
            }
            DecodeFn::AcFirst => {
                if self.eobrun > 0 {
                    self.eobrun -= 1;
                    return Ok(());
                }
                let (mut k, e) = (self.spectral_start, self.spectral_end);
                while k <= e {
                    let rs = self.decode_huffman(tables, c.huffman_ac)?;
                    let (s, r) = (rs & 15, rs >> 4);
                    if s == 0 {
                        if r < 15 {
                            self.eobrun = self.receive(r)? + (1i64 << r) - 1;
                            break;
                        }
                        k += 16;
                        continue;
                    }
                    k += r as usize;
                    let value = self.receive_and_extend(s)? * (1i64 << self.successive);
                    if let Some(&z) = DCT_ZIG_ZAG.get(k) {
                        c.set(offset + z as usize, value);
                    }
                    k += 1;
                }
            }
            DecodeFn::AcSuccessive => {
                let (mut k, e) = (self.spectral_start, self.spectral_end);
                let mut r: i64 = 0;
                while k <= e {
                    let z = DCT_ZIG_ZAG.get(k).map(|&z| offset + z as usize);
                    let current = z.map_or(0, |z| c.get(z));
                    let refine = |scan: &mut Self, c: &mut Component| -> Result<()> {
                        let bit = (scan.read_bit()? as i32) << scan.successive;
                        if let Some(z) = z {
                            c.set(z, current + bit as i64);
                        }
                        Ok(())
                    };
                    match self.successive_ac_state {
                        0 => {
                            let rs = self.decode_huffman(tables, c.huffman_ac)?;
                            let s = rs & 15;
                            r = rs >> 4;
                            if s == 0 {
                                if r < 15 {
                                    self.eobrun = self.receive(r)? + (1i64 << r);
                                    self.successive_ac_state = 4;
                                } else {
                                    r = 16;
                                    self.successive_ac_state = 1;
                                }
                            } else {
                                if s != 1 {
                                    bail!("invalid ACn encoding");
                                }
                                self.successive_ac_next_value = self.receive_and_extend(s)?;
                                self.successive_ac_state = if r != 0 { 2 } else { 3 };
                            }
                            continue;
                        }
                        1 | 2 => {
                            if current != 0 {
                                refine(self, c)?;
                            } else {
                                r -= 1;
                                if r == 0 {
                                    self.successive_ac_state = if self.successive_ac_state == 2 { 3 } else { 0 };
                                }
                            }
                        }
                        3 => {
                            if current != 0 {
                                refine(self, c)?;
                            } else {
                                if let Some(z) = z {
                                    c.set(z, (i32w(self.successive_ac_next_value) << self.successive) as i64);
                                }
                                self.successive_ac_state = 0;
                            }
                        }
                        _ => {
                            if current != 0 {
                                refine(self, c)?;
                            }
                        }
                    }
                    k += 1;
                }
                if self.successive_ac_state == 4 {
                    self.eobrun -= 1;
                    if self.eobrun == 0 {
                        self.successive_ac_state = 0;
                    }
                }
            }
        }
        Ok(())
    }
}

fn block_offset(c: &Component, row: usize, col: usize) -> usize {
    64 * ((c.blocks_per_line + 1) * row + col)
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    data: &[u8],
    offset: usize,
    frame: &mut Frame,
    selected: &[usize],
    reset_interval: usize,
    spectral_start: usize,
    spectral_end: usize,
    successive_prev: u32,
    successive: u32,
    tables: &[Huffman],
) -> Result<usize> {
    let mcus_per_line = frame.mcus_per_line;
    let f = if frame.progressive {
        match (spectral_start == 0, successive_prev == 0) {
            (true, true) => DecodeFn::DcFirst,
            (true, false) => DecodeFn::DcSuccessive,
            (false, true) => DecodeFn::AcFirst,
            (false, false) => DecodeFn::AcSuccessive,
        }
    } else {
        DecodeFn::Baseline
    };
    let mut scan = Scan {
        data,
        offset,
        bits_data: 0,
        bits_count: 0,
        eobrun: 0,
        successive_ac_state: 0,
        successive_ac_next_value: 0,
        spectral_start,
        spectral_end,
        successive,
    };
    let mcu_expected = if selected.len() == 1 {
        let c = &frame.components[selected[0]];
        c.blocks_per_line * c.blocks_per_column
    } else {
        mcus_per_line * frame.mcus_per_column
    };
    let reset_interval = if reset_interval == 0 { mcu_expected } else { reset_interval };
    let mut mcu = 0;
    while mcu < mcu_expected {
        for &i in selected {
            frame.components[i].pred = 0;
        }
        scan.eobrun = 0;
        if selected.len() == 1 {
            let c = &mut frame.components[selected[0]];
            for _ in 0..reset_interval {
                // `decodeBlock`; a zero `blocksPerLine` divides to NaN, an
                // index that writes nothing.
                if c.blocks_per_line == 0 {
                    bail!("component without blocks");
                }
                let offset = block_offset(c, mcu / c.blocks_per_line, mcu % c.blocks_per_line);
                scan.decode(f, c, offset, tables)?;
                mcu += 1;
            }
        } else {
            for _ in 0..reset_interval {
                for &i in selected {
                    let c = &mut frame.components[i];
                    for j in 0..c.v {
                        for k in 0..c.h {
                            let block_row = (mcu / mcus_per_line) * c.v + j;
                            let block_col = (mcu % mcus_per_line) * c.h + k;
                            let offset = block_offset(c, block_row, block_col);
                            scan.decode(f, c, offset, tables)?;
                        }
                    }
                }
                mcu += 1;
            }
        }
        scan.bits_count = 0;
        let marker = (byte(data, scan.offset) << 8) | byte(data, scan.offset + 1);
        if marker <= 0xFF00 {
            bail!("marker was not found");
        }
        if (0xFFD0..=0xFFD7).contains(&marker) {
            scan.offset += 2;
        } else {
            break;
        }
    }
    Ok(scan.offset - offset)
}

fn quantize_and_inverse(c: &mut Component, base: usize, p: &mut [i16; 64]) {
    let qt = c.quantization_table;
    for row in (0..64).step_by(8) {
        let raw: [i64; 8] = std::array::from_fn(|i| c.get(base + row + i));
        let [mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7] = raw;
        p0 *= qt[row] as i64;
        if (i32w(p1) | i32w(p2) | i32w(p3) | i32w(p4) | i32w(p5) | i32w(p6) | i32w(p7)) == 0 {
            let t = sar(DCT_SQRT2 * p0 + 512, 10) as i16;
            p[row..row + 8].fill(t);
            continue;
        }
        p1 *= qt[row + 1] as i64;
        p2 *= qt[row + 2] as i64;
        p3 *= qt[row + 3] as i64;
        p4 *= qt[row + 4] as i64;
        p5 *= qt[row + 5] as i64;
        p6 *= qt[row + 6] as i64;
        p7 *= qt[row + 7] as i64;
        let mut v0 = sar(DCT_SQRT2 * p0 + 128, 8);
        let mut v1 = sar(DCT_SQRT2 * p4 + 128, 8);
        let mut v2 = p2;
        let mut v3 = p6;
        let mut v4 = sar(DCT_SQRT1D2 * (p1 - p7) + 128, 8);
        let mut v7 = sar(DCT_SQRT1D2 * (p1 + p7) + 128, 8);
        let mut v5 = i32w(p3).wrapping_shl(4) as i64;
        let mut v6 = i32w(p5).wrapping_shl(4) as i64;
        v0 = sar(v0 + v1 + 1, 1);
        v1 = v0 - v1;
        let mut t = sar(v2 * DCT_SIN6 + v3 * DCT_COS6 + 128, 8);
        v2 = sar(v2 * DCT_COS6 - v3 * DCT_SIN6 + 128, 8);
        v3 = t;
        v4 = sar(v4 + v6 + 1, 1);
        v6 = v4 - v6;
        v7 = sar(v7 + v5 + 1, 1);
        v5 = v7 - v5;
        v0 = sar(v0 + v3 + 1, 1);
        v3 = v0 - v3;
        v1 = sar(v1 + v2 + 1, 1);
        v2 = v1 - v2;
        t = sar(v4 * DCT_SIN3 + v7 * DCT_COS3 + 2048, 12);
        v4 = sar(v4 * DCT_COS3 - v7 * DCT_SIN3 + 2048, 12);
        v7 = t;
        t = sar(v5 * DCT_SIN1 + v6 * DCT_COS1 + 2048, 12);
        v5 = sar(v5 * DCT_COS1 - v6 * DCT_SIN1 + 2048, 12);
        v6 = t;
        p[row] = (v0 + v7) as i16;
        p[row + 7] = (v0 - v7) as i16;
        p[row + 1] = (v1 + v6) as i16;
        p[row + 6] = (v1 - v6) as i16;
        p[row + 2] = (v2 + v5) as i16;
        p[row + 5] = (v2 - v5) as i16;
        p[row + 3] = (v3 + v4) as i16;
        p[row + 4] = (v3 - v4) as i16;
    }
    for col in 0..8 {
        let [p0, p1, p2, p3, p4, p5, p6, p7]: [i64; 8] = std::array::from_fn(|i| p[col + 8 * i] as i64);
        if (p1 | p2 | p3 | p4 | p5 | p6 | p7) == 0 {
            let mut t = sar(DCT_SQRT2 * p0 + 8192, 14);
            t = if t < -2040 {
                0
            } else if t >= 2024 {
                255
            } else {
                sar(t + 2056, 4)
            };
            for i in 0..8 {
                c.set(base + col + 8 * i, t);
            }
            continue;
        }
        let mut v0 = sar(DCT_SQRT2 * p0 + 2048, 12);
        let mut v1 = sar(DCT_SQRT2 * p4 + 2048, 12);
        let mut v2 = p2;
        let mut v3 = p6;
        let mut v4 = sar(DCT_SQRT1D2 * (p1 - p7) + 2048, 12);
        let mut v7 = sar(DCT_SQRT1D2 * (p1 + p7) + 2048, 12);
        let mut v5 = p3;
        let mut v6 = p5;
        v0 = sar(v0 + v1 + 1, 1) + 4112;
        v1 = v0 - v1;
        let mut t = sar(v2 * DCT_SIN6 + v3 * DCT_COS6 + 2048, 12);
        v2 = sar(v2 * DCT_COS6 - v3 * DCT_SIN6 + 2048, 12);
        v3 = t;
        v4 = sar(v4 + v6 + 1, 1);
        v6 = v4 - v6;
        v7 = sar(v7 + v5 + 1, 1);
        v5 = v7 - v5;
        v0 = sar(v0 + v3 + 1, 1);
        v3 = v0 - v3;
        v1 = sar(v1 + v2 + 1, 1);
        v2 = v1 - v2;
        t = sar(v4 * DCT_SIN3 + v7 * DCT_COS3 + 2048, 12);
        v4 = sar(v4 * DCT_COS3 - v7 * DCT_SIN3 + 2048, 12);
        v7 = t;
        t = sar(v5 * DCT_SIN1 + v6 * DCT_COS1 + 2048, 12);
        v5 = sar(v5 * DCT_COS1 - v6 * DCT_SIN1 + 2048, 12);
        v6 = t;
        let out = [v0 + v7, v1 + v6, v2 + v5, v3 + v4, v3 - v4, v2 - v5, v1 - v6, v0 - v7];
        for (i, &x) in out.iter().enumerate() {
            let x = if x < 16 {
                0
            } else if x >= 4080 {
                255
            } else {
                sar(x, 4)
            };
            c.set(base + col + 8 * i, x);
        }
    }
}

/// `Math.ceil` of `a / b / c` as JavaScript evaluates it.
fn ceil_div(a: usize, b: f64, c: usize) -> usize {
    (a as f64 / b / c as f64).ceil() as usize
}

/// `parse` then `getData(width, height, false)`.
pub fn decode(data: &[u8]) -> Result<Decoded> {
    let mut offset = 0usize;
    let read_u16 = |offset: &mut usize| {
        let v = (byte(data, *offset) << 8) | byte(data, *offset + 1);
        *offset += 2;
        v
    };
    let mut frame: Option<Frame> = None;
    let mut reset_interval = 0usize;
    let mut quantization_tables: Vec<Option<[u16; 64]>> = vec![None; 16];
    let mut huffman_dc: Vec<Option<usize>> = vec![None; 16];
    let mut huffman_ac: Vec<Option<usize>> = vec![None; 16];
    // Built tables; the slots above and the components refer to them by index.
    let mut tables: Vec<Huffman> = vec![];

    if read_u16(&mut offset) != 0xFFD8 {
        bail!("SOI not found");
    }
    let mut marker = read_u16(&mut offset);
    while marker != 0xFFD9 {
        match marker {
            0xFFE0..=0xFFEF | 0xFFFE => {
                // `readDataBlock`; APP0/APP14 contents are not used here.
                let length = read_u16(&mut offset) as usize;
                let end = (offset + length.saturating_sub(2)).min(data.len());
                offset += end.saturating_sub(offset.min(data.len()));
            }
            0xFFDB => {
                let end = read_u16(&mut offset) as usize + offset - 2;
                while offset < end {
                    let spec = byte(data, offset);
                    offset += 1;
                    let mut table = [0u16; 64];
                    match spec >> 4 {
                        0 => {
                            for &z in &DCT_ZIG_ZAG {
                                table[z as usize] = byte(data, offset) as u16;
                                offset += 1;
                            }
                        }
                        1 => {
                            for &z in &DCT_ZIG_ZAG {
                                table[z as usize] = read_u16(&mut offset) as u16;
                            }
                        }
                        _ => bail!("DQT: invalid table spec"),
                    }
                    quantization_tables[(spec & 15) as usize] = Some(table);
                }
            }
            0xFFC0..=0xFFC2 => {
                if frame.is_some() {
                    bail!("Only single frame JPEGs supported");
                }
                read_u16(&mut offset);
                let progressive = marker == 0xFFC2;
                offset += 1; // precision
                let scan_lines = read_u16(&mut offset) as usize;
                let samples_per_line = read_u16(&mut offset) as usize;
                let count = byte(data, offset) as usize;
                offset += 1;
                let mut components = vec![];
                let mut component_ids = vec![None; 256];
                let (mut max_h, mut max_v) = (0, 0);
                for _ in 0..count {
                    let id = byte(data, offset) as usize;
                    let h = (byte(data, offset + 1) >> 4) as usize;
                    let v = (byte(data, offset + 1) & 15) as usize;
                    max_h = max_h.max(h);
                    max_v = max_v.max(v);
                    let q = byte(data, offset + 2) as usize;
                    components.push(Component {
                        h,
                        v,
                        quantization_table: quantization_tables.get(q).copied().flatten().unwrap_or([0; 64]),
                        blocks_per_line: 0,
                        blocks_per_column: 0,
                        block_data: vec![],
                        pred: 0,
                        huffman_dc: None,
                        huffman_ac: None,
                    });
                    component_ids[id] = Some(components.len() - 1);
                    offset += 3;
                }
                if max_h == 0 || max_v == 0 {
                    bail!("invalid sampling factors");
                }
                // `prepareComponents`.
                let mcus_per_line = ceil_div(samples_per_line, 8.0, max_h);
                let mcus_per_column = ceil_div(scan_lines, 8.0, max_v);
                for c in &mut components {
                    c.blocks_per_line = ((samples_per_line as f64 / 8.0).ceil() * c.h as f64 / max_h as f64).ceil() as usize;
                    c.blocks_per_column = ((scan_lines as f64 / 8.0).ceil() * c.v as f64 / max_v as f64).ceil() as usize;
                    let size = 64 * (mcus_per_column * c.v) * (mcus_per_line * c.h + 1);
                    c.block_data = vec![0; size];
                }
                frame = Some(Frame {
                    samples_per_line,
                    scan_lines,
                    progressive,
                    max_h,
                    max_v,
                    mcus_per_line,
                    mcus_per_column,
                    components,
                    component_ids,
                });
            }
            0xFFC4 => {
                let length = read_u16(&mut offset) as usize;
                let mut i = 2;
                while i < length {
                    let spec = byte(data, offset);
                    offset += 1;
                    let mut code_lengths = [0u8; 16];
                    let mut sum = 0;
                    for l in &mut code_lengths {
                        *l = byte(data, offset) as u8;
                        sum += *l as usize;
                        offset += 1;
                    }
                    let values: Vec<u8> = (0..sum).map(|j| byte(data, offset + j) as u8).collect();
                    offset += sum;
                    i += 17 + sum;
                    tables.push(Huffman::build(&code_lengths, &values)?);
                    let slot = if spec >> 4 == 0 { &mut huffman_dc } else { &mut huffman_ac };
                    slot[(spec & 15) as usize] = Some(tables.len() - 1);
                }
            }
            0xFFDD => {
                read_u16(&mut offset);
                reset_interval = read_u16(&mut offset) as usize;
            }
            0xFFDA => {
                read_u16(&mut offset);
                let frame = frame.as_mut().ok_or_else(|| anyhow!("scan before frame"))?;
                let count = byte(data, offset) as usize;
                offset += 1;
                let mut selected = vec![];
                for _ in 0..count {
                    let index = frame.component_ids[byte(data, offset) as usize]
                        .ok_or_else(|| anyhow!("scan of an unknown component"))?;
                    let spec = byte(data, offset + 1) as usize;
                    offset += 2;
                    let c = &mut frame.components[index];
                    c.huffman_dc = huffman_dc.get(spec >> 4).copied().flatten();
                    c.huffman_ac = huffman_ac.get(spec & 15).copied().flatten();
                    selected.push(index);
                }
                let spectral_start = byte(data, offset) as usize;
                let spectral_end = byte(data, offset + 1) as usize;
                let approximation = byte(data, offset + 2);
                offset += 3;
                offset += decode_scan(
                    data,
                    offset,
                    frame,
                    &selected,
                    reset_interval,
                    spectral_start,
                    spectral_end,
                    approximation >> 4,
                    approximation & 15,
                    &tables,
                )?;
            }
            0xFFFF => {
                if byte(data, offset) != 255 {
                    offset -= 1;
                }
            }
            _ => {
                if offset >= 3
                    && byte(data, offset - 3) == 255
                    && (192..=254).contains(&byte(data, offset - 2))
                {
                    offset -= 3;
                } else {
                    bail!("unknown JPEG marker {marker:x}");
                }
            }
        }
        marker = read_u16(&mut offset);
    }
    let mut frame = frame.ok_or_else(|| anyhow!("JPEG without a frame"))?;

    // `buildComponentData`.
    let mut buffer = [0i16; 64];
    for c in &mut frame.components {
        for row in 0..c.blocks_per_column {
            for col in 0..c.blocks_per_line {
                let offset = block_offset(c, row, col);
                quantize_and_inverse(c, offset, &mut buffer);
            }
        }
    }

    // `_getLinearizedBlockData(this.width, this.height)`.
    let (width, height) = (frame.samples_per_line, frame.scan_lines);
    let n = frame.components.len();
    let mut out = vec![0u8; width * height * n];
    let mut x_offsets = vec![0u32; width];
    for (i, c) in frame.components.iter().enumerate() {
        let scale_x = c.h as f64 / frame.max_h as f64 * 1.0;
        let scale_y = c.v as f64 / frame.max_v as f64 * 1.0;
        let per_scanline = ((c.blocks_per_line + 1) << 3) as i64;
        for (x, o) in x_offsets.iter_mut().enumerate() {
            let j = (x as f64 * scale_x) as i32;
            *o = (((j & -8) << 3) | (j & 7)) as u32;
        }
        let mut offset = i;
        for y in 0..height {
            let j = (y as f64 * scale_y) as i32;
            let index = i32w(per_scanline * (j & -8) as i64) | ((j & 7) << 3);
            for &xo in &x_offsets {
                let at = index as i64 + xo as i64;
                out[offset] = if at >= 0 { c.block_data.get(at as usize).map_or(0, |&v| v as u8) } else { 0 };
                offset += n;
            }
        }
    }
    if n > 4 {
        bail!("Unsupported color mode");
    }
    if n == 3 {
        // `_convertYccToRgb`; the `Uint8Array` store truncates.
        let clamp = |a: f64| if a <= 0.0 { 0 } else if a >= 255.0 { 255 } else { a as u8 };
        for px in out.chunks_exact_mut(3) {
            let (y, cb, cr) = (px[0] as f64, px[1] as f64, px[2] as f64);
            px[0] = clamp(y - 179.456 + 1.402 * cr);
            px[1] = clamp(y + 135.459 - 0.344 * cb - 0.714 * cr);
            px[2] = clamp(y - 226.816 + 1.772 * cb);
        }
    }
    // Four components (CMYK / YCCK) are converted by jpgjs, but neuroglancer
    // rejects them, so they are returned as decoded.
    Ok(Decoded {
        width,
        height,
        num_components: n,
        data: out,
    })
}
