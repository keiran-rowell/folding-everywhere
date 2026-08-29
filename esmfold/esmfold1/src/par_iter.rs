//! Transparent parallel / sequential iterator abstraction for native vs WASM.

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub use rayon::prelude::*;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub trait ParChunksMutShim<'a, T: 'a> {
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T>;
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
impl<'a, T: 'a> ParChunksMutShim<'a, T> for &'a mut [T] {
    #[inline]
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T> {
        self.chunks_mut(chunk_size)
    }
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
impl<'a, T: 'a> ParChunksMutShim<'a, T> for &'a mut Vec<T> {
    #[inline]
    fn par_chunks_mut(self, chunk_size: usize) -> std::slice::ChunksMut<'a, T> {
        self.as_mut_slice().chunks_mut(chunk_size)
    }
}
