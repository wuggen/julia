use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::sync::GpuFuture;

use palette::{LinSrgba, Srgba};

use crate::{JuliaContext, JuliaData};

use std::cell::Cell;
use std::fmt::{self, Debug, Formatter};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImgDimensions {
    pub width: u32,
    pub height: u32,
}

pub struct JuliaExport {
    cached_data: Cell<Option<JuliaExportCache>>,
}

struct JuliaExportCache {
    dims: ImgDimensions,
    data: JuliaData,
    command_buffer: Arc<AutoCommandBuffer>,
    output_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl JuliaExport {
    pub fn new() -> JuliaExport {
        JuliaExport {
            cached_data: Cell::new(None),
        }
    }

    fn regen_cache(&self, dims: ImgDimensions, data: &JuliaData, context: &JuliaContext) {
        let shader_data = data.into_shader_data();
        let (input_buffer, future) =
            ImmutableBuffer::from_data(shader_data, BufferUsage::all(), context.queue().clone())
                .unwrap();

        let image = StorageImage::new(
            context.device().clone(),
            Dimensions::Dim2d {
                width: dims.width,
                height: dims.height,
            },
            Format::R8G8B8A8Unorm,
            Some(context.queue().family()),
        )
        .unwrap();

        let output_buffer = CpuAccessibleBuffer::from_iter(
            context.device().clone(),
            BufferUsage::all(),
            (0..dims.width * dims.height * 4).map(|_| 0u8),
        )
        .unwrap();

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(context.pipeline().clone(), 0)
                .add_image(image.clone())
                .unwrap()
                .add_buffer(input_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let command_buffer = Arc::new(
            AutoCommandBufferBuilder::primary(
                context.device().clone(),
                context.queue().family(),
            )
            .unwrap()
            .dispatch(
                [dims.width / 8, dims.height / 8, 1],
                context.pipeline().clone(),
                descriptor_set.clone(),
                (),
            )
            .unwrap()
            .copy_image_to_buffer(image.clone(), output_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
        );

        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.cached_data.set(Some(JuliaExportCache {
            dims,
            data: *data,
            command_buffer,
            output_buffer,
        }));
    }

    pub fn export(
        &self,
        dims: ImgDimensions,
        data: &JuliaData,
        filename: &Path,
        context: &JuliaContext,
    ) {
        let cache_opt = self.cached_data.take();

        match cache_opt {
            None => self.regen_cache(dims, data, context),
            Some(c) => {
                if c.data != *data || c.dims != dims {
                    self.regen_cache(dims, data, context);
                } else {
                    self.cached_data.set(Some(c));
                }
            }
        }

        self.export_core(filename, context);
    }

    fn export_core(&self, filename: &Path, context: &JuliaContext) {
        let cache = self.cached_data.take().unwrap();
        //self.command_buffer(context)
        cache
            .command_buffer
            .clone()
            .execute(context.queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let img_contents = cache.output_buffer.read().unwrap();
        let mut img_vec = Vec::new();
        img_vec.extend_from_slice(&img_contents[..]);
        drop(img_contents);

        for c in img_vec.chunks_exact_mut(4) {
            let lin = LinSrgba::<f32>::from_format(LinSrgba::new(c[0], c[1], c[2], c[3]));
            let nonlin = Srgba::<u8>::from_format(Srgba::from_linear(lin));
            let (r, g, b, a) = nonlin.into_components();
            c.copy_from_slice(&[r, g, b, a]);
        }

        /*
        for subpx in &img_contents[..] {
            img_vec.extend_from_slice(&subpx.to_ne_bytes());
        }
        */

        /*
        let image = ImageBuffer::<Rgba<u16>, _>::from_raw(
            cache.dims.width,
            cache.dims.height,
            &img_contents[..],
        )
        .unwrap();
        image.save(filename).unwrap();
        */

        image::save_buffer(filename, img_vec.as_ref(), cache.dims.width, cache.dims.height,
            image::ColorType::RGBA(8)).unwrap();

        self.cached_data.set(Some(cache));
    }
}

impl Debug for JuliaExport {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let cache = self.cached_data.take();
        let res = f
            .debug_struct("JuliaExport")
            .field(
                "cached_data",
                &match cache {
                    None => String::from("None"),
                    Some(ref c) => format!("{:?}", c.data),
                },
            )
            .finish();
        self.cached_data.set(cache);
        res
    }
}
