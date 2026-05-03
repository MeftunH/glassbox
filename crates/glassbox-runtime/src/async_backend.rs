use std::future::Future;
use std::pin::Pin;

use glassbox_core::Tensor;

use crate::backend::Backend;
use crate::error::Result;

#[cfg(not(target_arch = "wasm32"))]
pub trait AsyncBackend: Backend {
    fn download_async<'a>(
        &'a self,
        tensor: &'a Tensor,
    ) -> Pin<Box<dyn Future<Output = Result<Tensor>> + 'a>>;
}

#[cfg(target_arch = "wasm32")]
pub trait AsyncBackend: Backend {
    fn download_async<'a>(
        &'a self,
        tensor: &'a Tensor,
    ) -> Pin<Box<dyn Future<Output = Result<Tensor>> + 'a>>;
}

#[cfg(feature = "cpu")]
impl AsyncBackend for crate::cpu::CpuBackend {
    fn download_async<'a>(
        &'a self,
        tensor: &'a Tensor,
    ) -> Pin<Box<dyn Future<Output = Result<Tensor>> + 'a>> {
        Box::pin(async move { Backend::download(self, tensor) })
    }
}

#[cfg(feature = "wgpu")]
impl AsyncBackend for crate::wgpu_backend::WgpuBackend {
    fn download_async<'a>(
        &'a self,
        tensor: &'a Tensor,
    ) -> Pin<Box<dyn Future<Output = Result<Tensor>> + 'a>> {
        Box::pin(crate::wgpu_backend::WgpuBackend::download_async(self, tensor))
    }
}
