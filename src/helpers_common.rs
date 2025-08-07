use nohash_hasher::IntSet;
use rapidhash::v3::rapidhash_v3;

pub(crate) fn sort_pair<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn hash_intset(set: &IntSet<usize>) -> u64 {
    let mut sorted: Vec<_> = set.iter().copied().collect();
    sorted.sort_unstable();

    let mut buf = Vec::with_capacity(sorted.len() * std::mem::size_of::<usize>());
    for x in &sorted {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    rapidhash_v3(&buf)
}
