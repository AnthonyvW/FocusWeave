//! Raw array container used by the cross-check harness against OpenCV.

use crate::mat::{Img, Mat, MatU16, MatU8};
use std::fs;
use std::path::Path;

pub enum Any {
    U8(MatU8),
    U16(MatU16),
    F32(Mat),
}

pub fn read(path: impl AsRef<Path>) -> Any {
    let raw = fs::read(path).expect("read test array");
    let h = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let c = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let code = raw[12];
    let body = &raw[13..];
    match code {
        0 => Any::U8(Img::from_vec(h, w, c, body.to_vec())),
        2 => Any::U16(Img::from_vec(
            h,
            w,
            c,
            body.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        )),
        _ => Any::F32(Img::from_vec(
            h,
            w,
            c,
            body.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )),
    }
}

pub fn read_u8(path: impl AsRef<Path>) -> MatU8 {
    match read(path) {
        Any::U8(m) => m,
        _ => panic!("expected a uint8 array"),
    }
}

pub fn read_f32(path: impl AsRef<Path>) -> Mat {
    match read(path) {
        Any::F32(m) => m,
        Any::U8(m) => m.to_f32(),
        Any::U16(m) => Mat {
            h: m.h,
            w: m.w,
            c: m.c,
            data: m.data.iter().map(|v| f32::from(*v)).collect(),
        },
    }
}

fn header(h: usize, w: usize, c: usize, code: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(13);
    out.extend_from_slice(&(h as u32).to_le_bytes());
    out.extend_from_slice(&(w as u32).to_le_bytes());
    out.extend_from_slice(&(c as u32).to_le_bytes());
    out.push(code);
    out
}

pub fn write_f32(path: impl AsRef<Path>, m: &Mat) {
    let mut out = header(m.h, m.w, m.c, 1);
    for v in &m.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out).expect("write test array");
}

pub fn write_u8(path: impl AsRef<Path>, m: &MatU8) {
    let mut out = header(m.h, m.w, m.c, 0);
    out.extend_from_slice(&m.data);
    fs::write(path, out).expect("write test array");
}

pub fn write_u16(path: impl AsRef<Path>, m: &MatU16) {
    let mut out = header(m.h, m.w, m.c, 2);
    for v in &m.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out).expect("write test array");
}
