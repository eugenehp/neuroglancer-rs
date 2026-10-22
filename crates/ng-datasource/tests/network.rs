//! Public datasets over the network. Ignored by default:
//! `cargo test -p ng-datasource --test network -- --ignored`.

use futures::executor::block_on;
use ng_datasource::chunk::read_chunk;

/// FIB-25 (precomputed, JPEG chunks on `gs://`): the README's example. The
/// centre chunk of the coarsest scale, an edge chunk, decodes to its clipped
/// size. (Its bytes were checked against neuroglancer's jpgjs by hand: identical.)
#[test]
#[ignore]
fn fib25_jpeg_chunks_decode() {
    let source = block_on(ng_datasource::url::get("precomputed://gs://neuroglancer-public-data/flyem_fib-25/image")).unwrap();
    let volume = source.volume().unwrap();
    let scales = &volume.sources[0];
    let coarsest = scales.last().unwrap();
    let grid: Vec<u64> = coarsest.spec.grid_shape().iter().map(|&n| n / 2).collect();
    let chunk = block_on(read_chunk(coarsest, &grid)).unwrap();
    let numel: usize = chunk.size.iter().map(|&s| s as usize).product();
    assert_eq!(chunk.data.len(), numel);
    assert_eq!(chunk.size[..3], [36, 39, 62]);
    assert!(chunk.data.iter().any(|&v| v != 0));
    eprintln!("{} scales; chunk {grid:?} of the coarsest: size {:?}", scales.len(), chunk.size);
}

/// Hemibrain EM (precomputed, sharded JPEG with an identity hash and gzip
/// minishard indexes): centre chunks of the coarsest and a fine scale decode.
#[test]
#[ignore]
fn hemibrain_sharded_jpeg_chunks_decode() {
    let source = block_on(ng_datasource::url::get(
        "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
    ))
    .unwrap();
    let scales = &source.volume().unwrap().sources[0];
    for scale in [&scales[scales.len() - 1], &scales[2]] {
        let grid: Vec<u64> = scale.spec.grid_shape().iter().map(|&n| n / 2).collect();
        let chunk = block_on(read_chunk(scale, &grid)).unwrap();
        let numel: usize = chunk.size.iter().map(|&s| s as usize).product();
        assert_eq!(chunk.data.len(), numel);
        let nonzero = chunk.data.iter().filter(|&&v| v != 0).count();
        eprintln!("chunk {grid:?}: size {:?}, {nonzero} of {numel} nonzero", chunk.size);
        assert!(nonzero > 0);
    }
}
