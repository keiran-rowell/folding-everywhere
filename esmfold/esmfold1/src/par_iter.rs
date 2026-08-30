//! Transparent parallel / sequential iterator abstraction for native vs WASM.

#[cfg(any(not(any(target_arch = "wasm32", target_arch = "wasm64")), target_feature = "atomics"))]
pub use rayon::prelude::*;

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), not(target_feature = "atomics")))]
pub trait ParChunksMutShim<'a, T: 'a> {
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T>;
}

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), not(target_feature = "atomics")))]
impl<'a, T: 'a> ParChunksMutShim<'a, T> for &'a mut [T] {
    #[inline]
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T> {
        self.chunks_mut(chunk_size)
    }
}

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), not(target_feature = "atomics")))]
impl<'a, T: 'a> ParChunksMutShim<'a, T> for &'a mut Vec<T> {
    #[inline]
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T> {
        self.as_mut_slice().chunks_mut(chunk_size)
    }
}
