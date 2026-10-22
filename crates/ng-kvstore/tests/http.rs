//! `HttpStore` against a minimal local HTTP server that behaves like the
//! servers neuroglancer has to cope with.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use futures::executor::block_on;
use ng_kvstore::{parse_url, ByteRange};

const BODY: &[u8] = b"0123456789abcdef";

/// Serves: `/v` (honours `Range`), `/norange` (ignores it), `/forbidden`
/// (403), `/dir` (redirects to `/dir/`); anything else 404.
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let mut range = None;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("range: bytes=") {
                    let (a, b) = v.trim().split_once('-').unwrap();
                    range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()));
                }
            }
            let mut parts = request.split_whitespace();
            let (method, path) = (parts.next().unwrap(), parts.next().unwrap());
            let reply = |status: &str, headers: &str, body: &[u8]| {
                let body = if method == "HEAD" { &[][..] } else { body };
                format!("HTTP/1.1 {status}\r\n{headers}connection: close\r\n\r\n").into_bytes().into_iter().chain(body.iter().copied()).collect::<Vec<u8>>()
            };
            let out = match path {
                "/v" => match range {
                    Some((a, b)) => reply(
                        "206 Partial Content",
                        &format!("content-length: {}\r\ncontent-range: bytes {a}-{b}/{}\r\n", b - a + 1, BODY.len()),
                        &BODY[a..=b],
                    ),
                    None => reply("200 OK", &format!("content-length: {}\r\n", BODY.len()), BODY),
                },
                "/norange" => reply("200 OK", &format!("content-length: {}\r\n", BODY.len()), BODY),
                "/forbidden" => reply("403 Forbidden", "content-length: 0\r\n", b""),
                "/dir" => reply("301 Moved", &format!("location: http://{addr}/dir/\r\ncontent-length: 0\r\n"), b""),
                "/dir/" => reply("200 OK", "content-length: 7\r\n", b"listing"),
                _ => reply("404 Not Found", "content-length: 0\r\n", b""),
            };
            stream.write_all(&out).unwrap();
        }
    });
    format!("http://{addr}/")
}

#[test]
fn reads_like_neuroglancer() {
    let base = parse_url(&serve()).unwrap();
    let read = |k: &str, r| block_on(base.read(k, r)).unwrap();
    assert_eq!(read("v", ByteRange::Full).unwrap(), BODY);
    assert_eq!(read("v", ByteRange::Range { offset: 3, length: 4 }).unwrap(), b"3456");
    // Suffix: HEAD for the length, then a plain range request.
    assert_eq!(read("v", ByteRange::Suffix(5)).unwrap(), b"bcdef");
    // A server that ignores Range: sliced locally.
    assert_eq!(read("norange", ByteRange::Range { offset: 10, length: 3 }).unwrap(), b"abc");
    // Not found: 404, 403, and a redirect to a directory listing.
    assert_eq!(read("missing", ByteRange::Full), None);
    assert_eq!(read("forbidden", ByteRange::Full), None);
    assert_eq!(read("dir", ByteRange::Full), None);
}
